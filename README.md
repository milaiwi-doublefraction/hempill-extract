# hempill-extract


Assuming that the directory that contains the time period folders following
the structure below is in the parent directory called `hempill`, run this 
command:

1951-1970/
├── IndexDocs.txt
├── IndexDocsReference.txt
├── IndexLegals.txt
├── IndexLegalsReference.txt
├── IndexParties.txt
├── IndexPartiesReference.txt
├── IndexParcelNumbers.txt
├── IndexParcelsReference.txt
├── IndexRelatedDocs.txt
├── IndexRelatedReference.txt
└── OPR/
    └── 1962/
        └── Mar/
            └── 24/
                ├── 1962-1-1284562.tif
                └── ...


```python extract.py -v --input_path ../hempill/ --output_path ./out```
