from bs4 import BeautifulSoup, SoupStrainer
import string as Stringg
import json


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

            for word in title:  # adding the id of the file to the postings of the words in title
                if word in inverted_index:
                    if new_id not in inverted_index[word]:
                        inverted_index[word].append(new_id)
                else:
                    inverted_index[word] = [new_id]

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

            for word in body:  # adding the id of the file to the postings of the words in body
                if word in inverted_index:
                    if new_id not in inverted_index[word]:
                        inverted_index[word].append(new_id)
                else:
                    inverted_index[word] = [new_id]

            # print(inverted_index)
            index_of_reuters_end = data.index('</REUTERS>')
            handled_article = data[:index_of_reuters_end]
            data = data[index_of_reuters_end + 11:]  # I deleted the handled data by using slicing.

    print(inverted_index)
    with open("sample.json", "w") as outfile:
        json.dump(inverted_index, outfile)