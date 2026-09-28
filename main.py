"""
SmartScan - Document Scanner
Reference: PyImageSearch document scanner tutorial (2014)
Original additions:
  - automatic document enhancement + automatic output saving
  - a "mode" selector: choose --mode document for a crisp black-and-white
    scan (best for text), or --mode photo to keep color and just boost
    contrast/sharpness (best for photos, artwork, or whiteboards where
    you don't want everything crushed to black and white).
  - multi-page scanning: pass several photos and they get combined,
    in order, into a single multi-page PDF, like a real scanner app.

Run:
    python document_scanner.py --image page1.jpg --mode document
    python document_scanner.py --image page1.jpg page2.jpg page3.jpg -o packet.pdf

The program:
1. Finds the edges of the document.
2. Detects the largest four-corner contour.
3. Straightens the document using a perspective transform.
4. Enhances the result differently depending on --mode.
5. Saves the result: a single image file, or a multi-page PDF if you
   passed more than one --image (or asked for a .pdf output directly).
"""

import argparse
import cv2
import numpy as np
from PIL import Image


def order_points(pts):
    """Return points in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")

    sums = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sums)]       # top-left
    rect[2] = pts[np.argmax(sums)]       # bottom-right

    differences = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(differences)]  # top-right
    rect[3] = pts[np.argmax(differences)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    """Transform a four-corner document into a straight top-down view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def enhance_as_document(warped):
    """
    Document mode: convert to grayscale and apply adaptive thresholding
    so text becomes crisp black-on-white, like a real scanner output.
    Adaptive thresholding (rather than one fixed brightness cutoff) copes
    with photos that have shadows or uneven lighting across the page.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    scanned = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    return scanned


def enhance_as_photo(warped):
    """
    Photo mode: keep the image in color, but boost local contrast using
    CLAHE (Contrast Limited Adaptive Histogram Equalization) so details
    pop without blowing out bright areas or crushing shadows. Then apply
    a light sharpening pass so edges look crisp, similar to what a phone
    camera's "enhance" filter does.
    """
    # Work in the LAB color space so we only adjust lightness (L),
    # leaving color information (A, B) untouched.
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    contrast_boosted = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Light unsharp-mask style sharpening.
    blurred = cv2.GaussianBlur(contrast_boosted, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(contrast_boosted, 1.5, blurred, -0.5, 0)

    return sharpened


def process_page(image_path, mode="document"):
    """
    Run the full scan pipeline on a single photo and return the
    (original, scanned) image pair as numpy arrays. Does not save
    or display anything, so it can be reused for single-page and
    multi-page (PDF) output.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    original = image.copy()

    # Resize only for faster contour detection.
    ratio = image.shape[0] / 600.0
    resized = cv2.resize(
        image,
        (int(image.shape[1] / ratio), 600)
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Find outlines in the edge image.
    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    document_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        # A document is expected to have four corners.
        if len(approximation) == 4:
            document_contour = approximation.reshape(4, 2)
            break

    if document_contour is None:
        raise ValueError(
            "I could not find a clear four-corner document. "
            "Try a photo with the entire paper visible and good lighting."
        )

    # Convert the contour coordinates back to the original image size.
    document_contour = document_contour * ratio

    # Straighten the document.
    warped = four_point_transform(original, document_contour)

    # Enhance differently depending on the selected mode.
    if mode == "document":
        scanned = enhance_as_document(warped)
    else:  # mode == "photo"
        scanned = enhance_as_photo(warped)

    return original, scanned


def save_single_image(scanned, output_path):
    cv2.imwrite(output_path, scanned)


def save_as_pdf(scanned_pages, output_path):
    """
    Combine multiple scanned pages (numpy arrays, grayscale or color)
    into a single multi-page PDF, in the order they were given.
    """
    pil_pages = []

    for scanned in scanned_pages:
        # PDF pages need color images, so convert grayscale
        # (document mode) pages to RGB before handing them to Pillow.
        if len(scanned.shape) == 2:
            rgb = cv2.cvtColor(scanned, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB)

        pil_pages.append(Image.fromarray(rgb))

    first_page, remaining_pages = pil_pages[0], pil_pages[1:]
    first_page.save(
        output_path,
        save_all=True,
        append_images=remaining_pages
    )


def show_preview(original, scanned, label="Result"):
    cv2.imshow("Original", original)
    cv2.imshow(label, scanned)
    print("Press any key in the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="SmartScan - automatic document scanner"
    )
    parser.add_argument(
        "-i", "--image",
        required=True,
        nargs="+",
        help="Path(s) to the photo(s) of the document. Pass more than "
             "one to combine them, in order, into a multi-page PDF."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file name. Defaults to scanned_output.png for a "
             "single page, or scanned_output.pdf for multiple pages."
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["document", "photo"],
        default="document",
        help="'document' for a crisp black-and-white scan, "
             "'photo' to keep color and boost contrast/sharpness"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the popup preview window (useful for multi-page runs)"
    )

    args = parser.parse_args()

    is_multi_page = len(args.image) > 1
    output_path = args.output or ("scanned_output.pdf" if is_multi_page else "scanned_output.png")

    try:
        pages = [process_page(path, args.mode) for path in args.image]

        if is_multi_page or output_path.lower().endswith(".pdf"):
            scanned_pages = [scanned for _, scanned in pages]
            save_as_pdf(scanned_pages, output_path)
            print(f"\nScan complete! ({len(pages)} page(s), mode: {args.mode})")
            print(f"Saved combined PDF to: {output_path}")
        else:
            original, scanned = pages[0]
            save_single_image(scanned, output_path)
            print(f"\nScan complete! (mode: {args.mode})")
            print(f"Saved result to: {output_path}")

        if not args.no_preview:
            # Preview just the first page so we don't spawn a
            # window per page during a multi-page run.
            show_preview(pages[0][0], pages[0][1], label=f"Scanned Result ({args.mode} mode)")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()