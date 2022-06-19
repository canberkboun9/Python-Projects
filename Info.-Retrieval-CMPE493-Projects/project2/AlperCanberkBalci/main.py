import string as Stringg
import json
import math


file_names = ["reut2-000.sgm", "reut2-001.sgm", "reut2-002.sgm", "reut2-003.sgm",
              "reut2-004.sgm", "reut2-005.sgm", "reut2-006.sgm", "reut2-007.sgm",
              "reut2-008.sgm", "reut2-009.sgm", "reut2-010.sgm", "reut2-011.sgm",
              "reut2-012.sgm", "reut2-013.sgm", "reut2-014.sgm", "reut2-015.sgm",
              "reut2-016.sgm", "reut2-017.sgm", "reut2-018.sgm", "reut2-019.sgm",
              "reut2-020.sgm", "reut2-021.sgm"]
strx = Stringg.punctuation
stop_words = "a,all,an,and,any,are,as,be,been,but,by,few,for,have,he,her,here,him,his,\
how,i,in,is,it,its,any,me,my,none,of,on,or,our,she,some,the,their,them,there,\
they,that,this,us,was,what,when,where,which".split(",")
inverted_index = dict()
vector_space = dict()

for i in range(1, 21579):
    vector_space[i] = {}  # assign empty dictionaries into the documents. {word: tf.idf score}


def list_duplicates_of(list_of_words, word_):  # finds the indexes of word in a list of words, i.e. finds duplicates
    start_at = -1
    locs = []
    while True:
        try:
            loc = list_of_words.index(word_, start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def punctuation_removal(text):
    for word in text:
        if word in strx:
            text = text.replace(word, "")

    return text


def stop_word_removal(text):
    word_list = text.split()
    for word in word_list:
        if word in stop_words:
            word_list.pop(word_list.index(word))

    return word_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    for file_name in file_names:
        with open(file_name, 'r', encoding="latin-1") as f:
            data = f.read()

        print(file_name)
        data = data.strip('<!DOCTYPE lewis SYSTEM "lewis.dtd">')
        while data:
            index_of_id = data.index("NEWID=")
            index_of_date = data.index("DATE")
            new_id = int(data[index_of_id:index_of_date].strip('NEWID="').strip('">\n<'))

            # print(new_id)
            if "<TITLE>" not in data:
                index_of_reuters_end = data.index('</REUTERS>')
                handled_article = data[:index_of_reuters_end]
                data = data[index_of_reuters_end + 11:]  # I deleted the handled data by using slicing.
                continue
            else:
                index_of_title = data.index("<TITLE>")
                index_of_title_end = data.index("</TITLE>")

            title = data[index_of_title + 7:index_of_title_end]
            title = title.lower()  # case folding
            title = punctuation_removal(title)  # punctuation removal for title
            title = stop_word_removal(title)  # title is now a list, containing the words in the title

           # for word in title:  # adding the id of the file to the postings of the words in title
            #    if word in inverted_index:
             #       if new_id not in inverted_index[word]:
              #          inverted_index[word].append(new_id)
               # else:
                #    inverted_index[word] = [new_id]

            if "<BODY>" not in data:
                index_of_reuters_end = data.index('</REUTERS>')
                handled_article = data[:index_of_reuters_end]
                data = data[index_of_reuters_end + 11:]  # I deleted the handled data by using slicing.
                continue
            else:
                index_of_title = data.index("<TITLE>")
                index_of_title_end = data.index("</TITLE>")

            index_of_body = data.index("<BODY>")
            index_of_body_end = data.index("</BODY>")

            body = data[index_of_body + 6:index_of_body_end].strip("\n Reuter\n&#3;")
            body = body.lower()  # case folding
            body = punctuation_removal(body)  # punctuation removal for title
            body = stop_word_removal(body)  # body is now a list, containing the words in the body

            title.extend(body)  # title is now data = title + body as a list concatenated, and preprocessed.

            # we can add them into the inverted index.
            indexes = {}  # dict = {word: list of indexes of this word in this document}
            # find the indexes for the words in this new_id document. put them into indexes{}.
            for word in title:
                if word not in indexes:  # not to calculate redundantly
                    indexes[word] = list_duplicates_of(title, word)

            # tuple = (new_id, indexes) is the element of the list of inverted_index[word]

            for word in title:  # adding the id of the file to the postings of the words in body
                check_list = []  # checking list
                if word in inverted_index:
                    for i in range(len(inverted_index[word])):
                        check_list.append(inverted_index[word][0])  # collect the doc_ids.

                    if new_id not in check_list:
                        inverted_index[word].append((new_id, indexes[word]))  # new_id, list of indexes as a tuple
                else:
                    inverted_index[word] = [(new_id, indexes[word])]  # new_id, list of indexes as a tuple

            set_of_words = set(title)
            set_of_words = list(set_of_words)
            for word in set_of_words:
                term_freq = title.count(word)
                if term_freq > 0:
                    term_freq = 1 + math.log(term_freq,10)  # Put the TF into the dict, later calculate tf.idf

                vector_space[new_id][word] = term_freq  # calculate tf.idf later because we cannot calculate idf now.
            # calculate tf.idf score for each unique word in the document and add it to the dictionary
            # vector_space[new_id][word] = tf.idf score value, then normalize.

            # print(inverted_index)
            index_of_reuters_end = data.index('</REUTERS>')
            handled_article = data[:index_of_reuters_end]
            data = data[index_of_reuters_end + 11:]  # I deleted the handled data by using slicing.

    # Calculating the tf.idf values and putting them into the vector_space
    for doc in range(1, 21579):
        sum_ = 0
        for word in vector_space[doc]:
            tf = vector_space[doc][word]
            df = len(inverted_index[word])  # not sure if it returns the document frequency
            x = 21578/df
            idf = math.log(x, 10)
            tf_idf = tf*idf  # tf.idf calculated
            vector_space[doc][word] = tf_idf  # not normalized tf.idf, I want to make the vector's length = 1.
            sum_ += tf_idf**2  # square of the value is taken to normalize later.

        vector_length = math.sqrt(sum_)
        for word in vector_space[doc]:
            vector_space[doc][word] = vector_space[doc][word]/vector_length  # normalized value.

    with open("sample.json", "w") as outfile:  # thanks to "with", file is closed after this.
        json.dump(inverted_index, outfile)

    with open("vector_space.json", "w") as outfile:
        json.dump(vector_space, outfile)