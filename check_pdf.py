from pathlib import Path
from pypdf import PdfReader

pdf_folder = Path("data/uploaded_pdfs")

pdfs = list(pdf_folder.glob("*.pdf"))

if not pdfs:
    print("No PDF found!")
    exit()

pdf_path = pdfs[0]

print("Checking:", pdf_path.name)

reader = PdfReader(pdf_path)

print("Pages:", len(reader.pages))

for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    print("=" * 60)
    print(f"Page {i}")

    if text:
        print("Characters:", len(text))
        print(text[:300])
    else:
        print("NO TEXT FOUND")