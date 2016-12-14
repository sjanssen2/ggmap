import sys
import types


def parse_metaphlan_markers_info(filename, abort_after_lines=20):
    clades = {}
    try:
        file = open(filename, 'r')
        readlines = 0
        for line in file:
            readlines += 1
            if line.startswith('gi|'):
                type_ids = 'gi'
                accession = (line.split('\t')[0]).split('|')[1]
            elif line.startswith('GeneID:'):
                type_ids = 'GeneID'
                accession = (line.split('\t')[0]).split(':')[1]
            elif line.startswith('NC_'):
                type_ids = 'NC'
                accession = line.split('\t')[0]
            else:
                type_ids = None
                accession = None

            if (type_ids is not None) and (accession is not None):
                clade = line.split("clade': '")[1].split("'")[0]
                if clade not in clades:
                    clades[clade] = {}
                if type_ids not in clades[clade]:
                    clades[clade][type_ids] = {}
                clades[clade][type_ids][accession] = True

            if (abort_after_lines is not None) and \
               (readlines >= abort_after_lines):
                break
        file.close()
        return clades

    except IOError:
        print('Cannot read file')

if __name__ == "__main__":
    abort_after_lines = None

    clades = (parse_metaphlan_markers_info('/home/sjanssen/GreenGenes/Metaphlan/markers_info.txt', abort_after_lines=abort_after_lines))
    x = 0
    for c in clades:
        x+=1
        type_ids = list(clades[c].keys())[0]
        print(c, clades[c][type_ids].keys())

    print("Ende")
