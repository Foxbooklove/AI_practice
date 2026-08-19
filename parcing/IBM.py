from docling.document_converter import DocumentConverter

conv = DocumentConverter()
result = conv.convert("paper.pdf")

print(result.document.export_to_markdown())