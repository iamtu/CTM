import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage: python convert_2_LDA.py input_file output_file'
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    ins = open(input_file, 'r')
    D = int(ins.readline().strip())
    W = int(ins.readline().strip())
    NNZ = int(ins.readline().strip())

    docs = {}
    for i in xrange(NNZ):
        line = ins.readline()
        [docID, wordID, count] = line.strip().split()
        docID = int(docID)
        wordID = int(wordID)
        count = int(count)

        if docID not in docs:
            docs[docID] = {}

        if wordID not in docs[docID]:
            docs[docID][wordID] = count
        else:
            docs[docID][wordID] += count

    ins.close()
    print 'finish readfile'

    with open(output_file, 'w') as out:
        for docID in docs:
            new_str = str(len(docs[docID])) + ' '
            for wordID in docs[docID]:
                new_str += str(wordID) + ':' + str(docs[docID][wordID]) + ' '
            new_str = new_str[:-1]
            out.write(new_str + '\n')

    print 'finish write'
