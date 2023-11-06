from pylatex import Command, Document, Section, Subsection, Figure
from pylatex.utils import NoEscape, italic
from pathlib import Path

class Report():
    def __init__(self, title:str, author:str = None):
        self.document = Document()

        self.document.preamble.append(Command("title", title))
        if author != None:
            self.document.preamble.append(Command("author", "Anonymous author"))
        self.document.preamble.append(Command("date", NoEscape(r"\today")))
        self.document.append(NoEscape(r"\maketitle"))



    def add_section(self, title: str, text: str):
        with self.document.create(Section(title)):
            self.document.append(text)


    def add_subsection(self, title: str, text: str):
        with self.document.create(Subsection(title)):
            self.document.append(text)

    def add_plot(self, width: int, caption: str, image_filename: str):
        with self.document.create(Figure(position="htbp")) as plot:
            #plot.add_plot(width=NoEscape(width))
            plot.add_image(image_filename, width=NoEscape(width))
            plot.add_caption(caption)

    def save_document(self, document_path: Path):
        self.document.generate_pdf(str(document_path.joinpath("EDA_REPORT")))