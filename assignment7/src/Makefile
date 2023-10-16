SHELL:=/bin/bash
HOSTNAME:=$(shell hostname)
BIN:=instructions-$(HOSTNAME)
SRC:=instructions.cpp
CSV_FILE:=$(BIN).csv

CXX:=g++
CXXFLAGS ?= -O3 -g -fno-tree-vectorize

OP?=+

all: $(BIN) branch

$(BIN): $(SRC) Makefile
	$(CXX) $(CXXFLAGS) -DOP=$(OP) $< -o $@

.PHONY: clean bench-instructions

bench-instructions:
	@echo dtype,op,chains,time > $(CSV_FILE)
	for o in + "*" /; do \
		make $(BIN) -B OP="$$o"; \
		./$(BIN) 27 >> $(CSV_FILE); \
	done;

clean:
	$(RM) -f $(BIN) branch
