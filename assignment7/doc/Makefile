TARGET = assignment07

all : settings pdf # dvi # pdf ps 

dvi  : $(TARGET).dvi
pdf  : $(TARGET).pdf
html : $(TARGET).html

settings : 
	@echo ==================================
	@echo "TARGET    = $(TARGET)"
	@echo "VPATH     = $(VPATH)"	
	@echo ==================================


$(TARGET).pdf : $(TARGET).tex 
	pdflatex $(TARGET).tex
	pdflatex $(TARGET).tex

$(TARGET).html : $(TARGET).tex
	hevea -fix $(TARGET).tex

$(TARGET).dvi : $(TARGET).tex 
	latex $(TARGET).tex
	latex $(TARGET).tex

clean :
	rm -f *.aux *.dvi *.ps *.log *.out *.bbl *.blg *~
	rm -f $(TARGET).tgz
	rm -f $(TARGET).pdf
	rm -f *.html *.gif *.haux *.htoc
	rm -f $(TARGET)*.png
	rm -f $(TARGET).synctex
	rm -f $(TARGET).image.tex
