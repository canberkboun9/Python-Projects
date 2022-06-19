import string as Stringg
import json
import math


json1_file = open('sample.json')
json1_str = json1_file.read()
inverted_index = json.loads(json1_str)

json2_file = open('vector_space.json')
json2_str = json2_file.read()
vector_space = json.loads(json2_str)

strx = Stringg.punctuation
stop_words = "a,all,an,and,any,are,as,be,been,but,by,few,for,have,he,her,here,him,his,\
how,i,in,is,it,its,any,me,my,none,of,on,or,our,she,some,the,their,them,there,\
they,that,this,us,was,what,when,where,which".split(",")


def intersect(index_counter, word1, word2):
    answer = []
    len_1 = len(word1)
    len_2 = len(word2)
    i = 0
    j = 0
    while i < len_1 and j < len_2:
        if int(word1[i][0]) == int(word2[j][0]):
            # print("I found ", word1[i][0])
            k = 0
            l = 0
            while k < len(word1[i][1]) and l < len(word2[j][1]):
                if int(word1[i][1][k]) + index_counter == int(word2[j][1][l]):  # look for via (+index_counter).
                    if word1[i] not in answer:
                        answer.append(word1[i])  # put the (new_id, [indexes]) into the answer list.

                    k += 1
                    l += 1
                elif int(word1[i][1][k]) + index_counter < int(word2[j][1][l]):
                    k += 1
                else:
                    l += 1

            i+=1
            j+=1
        elif int(word1[i][0]) < int(word2[j][0]):
            i+=1
        else:
            j+=1

    return answer

# def conjunction(word1, word2):
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


# Main

is_continue = "continue"

while is_continue != "q" and is_continue != "Q":
    print("Please enter the query: ")
    query = input()

    is_phrase_query = False
    if query[0] == '"' and query[-1] == '"':
        is_phrase_query = True
        query = query[1:-1]

    # query = x.split()
    words = []
    result = []

    if is_phrase_query:  # AND, that is, phrase query
        # preprocess: case folding, stop word and punctuation removal
        query = query.lower()  # case folding
        query = punctuation_removal(query)  # punctuation removal for query
        query = stop_word_removal(query)  # query is now a list, containing the words in the query

        # First check whether all the tokens in the query exist in the inverted_index.
        existence_check = True
        for word in query:
            if word not in inverted_index:
                existence_check = False

        if existence_check:
            # Check whether the length of the query < 2
            if len(query) == 0:
                print("Invalid words...")  # Try Again.
            elif len(query) == 1:
                for element in inverted_index[query[0]]:
                    result.append(element[0])  # collect the new_id values from the tuples in the list.

                # result = inverted_index[query[0]]
            else:  # len(query) >=2:
                # Recursive merge algorithm.
                result = inverted_index[query[0]]  # put the (new_id, [indexes of query[0]]) into result.
                # print(result)
                for i in range(1, len(query)):  # from 1 to len: if 3 words, iterates 2 times.
                    result = intersect(i, result, inverted_index[query[i]])

        # result variable consists of tuples, however, we need only 0th index of the tuples.
        # print(result)
        result_ids = []
        for tuple_ in result:
            # print(tuple_)
            result_ids.append(tuple_[0])

        print("Result: ", result_ids)

    else:  # free text query.

        # Full of OR query.

        # preprocess: case folding, stop word and punctuation removal
        query = query.lower()  # case folding
        query = punctuation_removal(query)  # punctuation removal for query
        query = stop_word_removal(query)  # query is now a list, containing the words in the query


        # We don't need the merge algorithm here.
        docs_containing_query_words = []
        for word in query:
            if word in inverted_index:
                # print(inverted_index[word])
                if inverted_index[word] not in result:
                    result.extend(inverted_index[word])
                # result.extend(list(set(inverted_index[word][0])))  # to avoid duplicates.

        for elem in result:
            if elem[0] not in docs_containing_query_words:
                docs_containing_query_words.append(elem[0])

        # Step 1: create the query vector, and normalize the query vector
        # treat the query vector as a document, create a dict to assume it a vector. {word: tf.idf value}
        query_dict = {}
        set_of_words = list(set(query))

        sum_ = 0
        for word in set_of_words:
            term_freq = query.count(word)
            if term_freq > 0:
                term_freq = 1 + math.log(term_freq, 10)  # Put the TF into the dict, later calculate tf.idf

            query_dict[word] = term_freq  # calculate tf.idf later because we cannot calculate idf now.
            if word in inverted_index:
                df = len(inverted_index[word]) + 1  # add 1 as query is calculated 1
            else:
                df = 1
            # calculate tf.idf score for each unique word in the document and add it to the dictionary
            x = 21578 / df
            idf = math.log(x, 10)
            tf_idf = term_freq * idf  # tf.idf calculated
            query_dict[word] = tf_idf  # not normalized tf.idf, I want to make the query vector's length = 1.
            sum_ += tf_idf ** 2  # square of the value is taken to normalize later.

        query_length = math.sqrt(sum_)
        for word in set_of_words:
            query_dict[word] = query_dict[word] / query_length  # normalized value.
        # Calculating the tf.idf values and putting them into the query_dict is done.

        # Take each document containing any word of query, calculate cosine similarity with the normalized query vector.
        cosine_similarity_dict = {}
        for doc in docs_containing_query_words:  # iterate the documents
            product = 0  # product is the cosine similarity element to be summed
            # print("doc id = ", doc)
            # print(vector_space[str(doc)])
            for word, tf_idf_of_word in vector_space[str(doc)].items():  # iterate the words of the doc, take its tf.idf
                for query_word in query:  # check if the word
                    if word == query_word:
                        product += tf_idf_of_word*query_dict[query_word]  # sum the products to obtain cosine similarity
            cosine_similarity_dict[doc] = product  # put the cosine similarity value into the dict, later sort it

        # Sort the list according to the cosine similarity, that is value of the dictionary key,value.
        cosine_sorted_list = {k: v for k, v in sorted(cosine_similarity_dict.items(), key=lambda item: item[1])}
        # print the sorted list.
        print("Result:")
        for key, value in cosine_sorted_list.items():
            print(key, ": ", value)
        # print("Result: ", cosine_sorted_list)  # docs değil sorted list.

    print("If you want to quit, enter the letter 'Q', If you want to continue, enter another letter: ")
    is_continue = input()

print("Have a nice day! Thank you for using the query processor master99.")

