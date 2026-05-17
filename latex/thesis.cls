\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[2018/03/25 Uet Graduation]
\LoadClass[oneside, a4paper]{report}

\RequirePackage{times}
\RequirePackage[fontsize=13pt]{scrextend}%thesis' requirements
\RequirePackage[utf8]{inputenc}
\RequirePackage[english, vietnamese]{babel}
\RequirePackage[a4paper, top=2.5cm, bottom=3cm, left=3cm, right=2cm]{geometry}%thesis' requirements
\RequirePackage{array}
\RequirePackage{titlesec}
\RequirePackage{etoolbox}%fix titlesec's bug
\RequirePackage{tocloft}
\RequirePackage{caption}
\RequirePackage[linktocpage=true, hidelinks]{hyperref}%page's link
\RequirePackage{lipsum}
\RequirePackage{graphicx}%images
\RequirePackage{tikz}
\RequirePackage{float}
\RequirePackage{amsmath}%math funcs
\RequirePackage{amssymb}
\RequirePackage{mathtools}
\RequirePackage{pgfplots}%plot
\RequirePackage{adjustbox}
\RequirePackage{listings}%code
\RequirePackage{algorithm2e}%pseudo code
\RequirePackage{booktabs}


%thesis' requirements
\renewcommand{\baselinestretch}{1.3}
\setlength{\parskip}{6pt}
\setlength{\parindent}{1cm}

%fix section counter bug from titlesec
\makeatletter
\patchcmd{\ttlh@hang}{\parindent\z@}{\parindent\z@\leavevmode}{}{}
\patchcmd{\ttlh@hang}{\noindent}{}{}{}
\makeatother

%table caption above
\floatstyle{plaintop}
\restylefloat{table}

%images folder
\graphicspath{ {images/} {pdf/} }

\usetikzlibrary{arrows.meta, matrix, calc, chains, positioning, decorations.pathreplacing, arrows}
