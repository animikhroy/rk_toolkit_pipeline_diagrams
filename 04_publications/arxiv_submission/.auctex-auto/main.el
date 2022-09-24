(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("biblatex" "backend=biber" "sorting=ynt")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "sections/abstract"
    "sections/introduction"
    "sections/formulaism"
    "sections/novel_approach"
    "sections/implementation"
    "sections/store_sales"
    "sections/ligo"
    "sections/conclusion"
    "article"
    "art10"
    "arxiv"
    "inputenc"
    "fontenc"
    "hyperref"
    "url"
    "booktabs"
    "amsfonts"
    "dirtytalk"
    "color"
    "listings"
    "newtxtt"
    "nicefrac"
    "microtype"
    "lipsum"
    "graphicx"
    "multicol"
    "amsmath"
    "cuted"
    "float"
    "biblatex"
    "titlesec")
   (LaTeX-add-bibliographies
    "bibs/ref"))
 :latex)

