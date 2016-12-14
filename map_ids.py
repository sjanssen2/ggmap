import sys
import types


def parse_gg_otu_map(filename, abort_after_lines=20):
    otus = {}
    try:
        file = open(filename, 'r')
        readlines = 0
        for line in file:
            readlines += 1
            fields = line.rstrip().split("\t")
            otus[int(fields[1])] = list(map(int, fields[2:]))
            if (abort_after_lines is not None) and \
               (readlines >= abort_after_lines):
                break
        file.close()
        return otus
    except IOError:
        print('Cannot read file')

# def parse_metaphlan_markers_info(filename, abort_after_lines=20):
#     clades = {}
#     try:
#         file = open(filename, 'r')
#         readlines = 0
#         for line in file:
#             readlines += 1
#             if line.startswith('gi|'):
#                 type_ids = 'gi'
#                 accession = (line.split('\t')[0]).split('|')[1]
#             elif line.startswith('GeneID:'):
#                 type_ids = 'GeneID'
#                 accession = (line.split('\t')[0]).split(':')[1]
#             elif line.startswith('NC_'):
#                 type_ids = 'NC'
#                 accession = line.split('\t')[0]
#             else:
#                 type_ids = None
#                 accession = None
#
#             if (type_ids is not None) and (accession is not None):
#                 clade = line.split("clade': '")[1].split("'")[0]
#                 if clade not in clades:
#                     clades[clade] = {}
#                 if type_ids not in clades[clade]:
#                     clades[clade][type_ids] = {}
#                 clades[clade][type_ids][accession] = True
#
#             if (abort_after_lines is not None) and \
#                (readlines >= abort_after_lines):
#                 break
#         file.close()
#         return clades
#
#     except IOError:
#         print('Cannot read file')
#
if __name__ == "__main__":
    abort_after_lines = None
    file_otumap97 = '/home/sjanssen/GreenGenes/gg_13_5_otus/otus/97_otu_map.txt'
    x = parse_gg_otu_map(file_otumap97, abort_after_lines)
    print([ [i]+x[i] for i in x if i in [3206355, 193610, 198101, 188187] ])
#
#     clades = (parse_metaphlan_markers_info('/home/sjanssen/GreenGenes/Metaphlan/markers_info.txt', abort_after_lines=abort_after_lines))
#     x = 0
#     for c in clades:
#         x+=1
#         type_ids = list(clades[c].keys())[0]
#         print(c, clades[c][type_ids].keys())
#
    print("Ende")
