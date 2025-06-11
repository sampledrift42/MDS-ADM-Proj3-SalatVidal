all: main.pdf

main.pdf: main.tex references.bib
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.pdf 