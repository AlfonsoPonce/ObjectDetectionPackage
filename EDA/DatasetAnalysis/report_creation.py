'''
This module contains the class to represent an automatic generated report. All document edit functions
must be placed here.

Author: Alfonso Ponce Navarro

Date: 08/11/2023
'''
from pylatex import Command, Document, Section, Subsection, Figure
from pylatex.utils import NoEscape, italic
from pathlib import Path


class Report():
    def __init__(self, title: str, author: str = None):
        '''
        Initialise report object with a document title and besides the author.

        :param title: Title for the report.
        :param author: Name of the report's author.
        '''
        self.document = Document()

        self.document.preamble.append(Command("title", title))
        if author is not None:
            self.document.preamble.append(
                Command("author", "Anonymous author"))
        self.document.preamble.append(Command("date", NoEscape(r"\today")))
        self.document.append(NoEscape(r"\maketitle"))

    def add_section(self, title: str, text: str) -> None:
        '''
        Function to add a section.

        :param title: Title of the section.
        :param text: Text to be put inside the section.
        :return:
        '''
        with self.document.create(Section(title)):
            self.document.append(text)

    def add_subsection(self, title: str, text: str) -> None:
        '''
        Function to add a subsection.

        :param title: Title of the subsection.
        :param text: Text to be put inside the subsection.
        '''
        with self.document.create(Subsection(title)):
            self.document.append(text)

    def add_plot(self, width: int, caption: str, image_filename: str) -> None:
        '''
        Function to add a plot to the report.

        :param width: width of the plot.
        :param caption: caption of the images.
        :param image_filename: Name of the image. Must end with image extension.
        '''
        with self.document.create(Figure(position="htbp")) as plot:
            # plot.add_plot(width=NoEscape(width))
            plot.add_image(image_filename, width=NoEscape(width))
            plot.add_caption(caption)

    def save_document(self, document_path: Path) -> None:
        '''
        Function to save the document into pdf. Name of the document must not be passed.

        :param document_path: Path where the document will be save, without it's name.
        '''
        self.document.generate_pdf(str(document_path.joinpath("EDA_REPORT")))
