Compile the manuscript with a standard TeX Live installation and pdfLaTeX.
The included ICLR style, bibliography style, and figure assets are required.

```sh
pdflatex -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
bibtex iclr2027_conference
pdflatex -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
pdflatex -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
pdflatex -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

The resulting PDF includes the main paper, statements, references, and appendices.
