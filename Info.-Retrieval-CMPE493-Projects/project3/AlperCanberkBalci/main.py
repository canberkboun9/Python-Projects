import string as Stringg
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
inverted_index_train = dict()
inverted_index_test = dict()
vector_space_train = dict()
vector_space_test = dict()
# classes_docs = dict()  # {'class1' : [doc1, doc2], 'class2' : [doc3, doc4]}, replaced with below topics dicts.
topics_train = {}  # holds the number of times the topic is used for an article
topics_test = {}  # holds the number of times the topic is used for an article
topics_train_len = {}
topics_test_len = {}
id_topics_training_set = {}
id_topics_test_set = {}
mega_doc = dict()  # {'class1' : {'word1' : count1, ..}, 'class2': {..}, ... }
# where count1 = n_k, num of occurences of w_k in textj(megadocj)
mega_doc_count = dict()  # {'class1' : n, ... } , where n is the number of tokens in textj(megadocj)
vocabulary = set()
testing_dict = dict()  # {'new_id' : ['word1', 'word2',...], 'new_id2' : [...],...}

for i in range(1, 21579):  # 6494 train set size.
    vector_space_train[i] = {}  # assign empty dictionaries into the documents. {word: tf.idf score}

for i in range(1, 21579):  # 2548 test set size.
    vector_space_test[i] = {}  # assign empty dictionaries into the documents. {word: tf.idf score}


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
    # First read all the articles to find out the most frequently used 10 classes #####

    for file_name_ in file_names:
        with open(file_name_, 'r', encoding="latin-1") as f:
            data = f.read()

        print(file_name_)
        data = data.strip('<!DOCTYPE lewis SYSTEM "lewis.dtd">')

        while data:
            train_or_test = ""
            index_of_train_or_test = data.index("LEWISSPLIT=")
            index_of_train_or_test_end = data.index("CGISPLIT")
            which = data[index_of_train_or_test+12:index_of_train_or_test_end-2]
            if which == "TEST":
                train_or_test = "test"
            elif which == "TRAIN":
                train_or_test = "train"
            else:
                train_or_test = "neither"

            index_of_topics = data.index("<TOPICS>")
            index_of_topics_end = data.index("</TOPICS>")
            topics_string = data[index_of_topics+8:index_of_topics_end]

            index_of_id = data.index("NEWID=")
            index_of_date = data.index("DATE")
            new_id = int(data[index_of_id:index_of_date].strip('NEWID="').strip('">\n<'))

            current_doc_topics = []
            while "<D>" in topics_string:
                index_of_D = topics_string.index("<D>")
                index_of_D_end = topics_string.index("</D>")
                topic_ = topics_string[index_of_D+3:index_of_D_end]
                current_doc_topics.append(topic_)
                if train_or_test == "train":
                    if topic_ in topics_train:
                        topics_train_len[topic_] += 1
                        topics_train[topic_].append(new_id)
                    else:
                        topics_train_len[topic_] = 1
                        topics_train[topic_] = [new_id]
                elif train_or_test == "test":
                    if topic_ in topics_test:
                        topics_test_len[topic_] += 1
                        topics_test[topic_].append(new_id)
                    else:
                        topics_test_len[topic_] = 1
                        topics_test[topic_] = [new_id]

                topics_string = topics_string[index_of_D_end+4:]  # erase the found topic and continue

            if train_or_test == "train":
                id_topics_training_set[new_id] = current_doc_topics
            elif train_or_test == "test":
                id_topics_test_set[new_id] = current_doc_topics

            index_of_reuters_end = data.index('</REUTERS>')
            # handled_article = data[:index_of_reuters_end]
            data = data[index_of_reuters_end + 11:]  # I deleted the handled data by using slicing.

    sorted_train = sorted(topics_train_len.items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_test = sorted(topics_test_len.items(), key=lambda x: x[1], reverse=True)[:10]
    print("sorted_train:", sorted_train)
    print("sorted_test:", sorted_test)

    max_ten_topics = []
    for topicAndValue in sorted_train:
        max_ten_topics.append(topicAndValue[0])

    set_topics_max_ten = set(max_ten_topics)

    sorted_id_topics_training_set = {}  # I will delete the useless documents and put them here.
    sorted_id_topics_test_set = {}

    for new_id_i, topics_i in id_topics_training_set.items():
        for topic_i in max_ten_topics:
            if topic_i in topics_i:
                set_topics_i = set(topics_i)
                intersexn = set_topics_i.intersection(set_topics_max_ten)
                sorted_id_topics_training_set[new_id_i] = list(intersexn)
                continue

    for new_id_i, topics_i in id_topics_test_set.items():
        for topic_i in max_ten_topics:
            if topic_i in topics_i:
                set_topics_i = set(topics_i)
                intersexn = set_topics_i.intersection(set_topics_max_ten)
                sorted_id_topics_test_set[new_id_i] = list(intersexn)
                continue

    id_list_training = list(sorted_id_topics_training_set.keys())
    id_list_test = list(sorted_id_topics_test_set.keys())
    # print(sorted_id_topics_training_set)

    # We will use these 10 topics for training.
    # Then, read all the articles that are in the most frequently used 10 classes. #####
    for topic_i in sorted_train:
        mega_doc[topic_i[0]] = {}  # open the dict as value of the key for the top ten classes in mega_doc dict.
        mega_doc_count[topic_i[0]] = 0  # holds the n parameter of the class_i.

    # print(len(sorted_id_topics_training_set))  # how many documents in test and train sets:
    # print(len(sorted_id_topics_test_set))  # for report part 2.
    # How many of these multilabelled (still part 2):
    '''multilabel_count = 0
    for value in list(sorted_id_topics_training_set.values()):
        if len(value) > 1:
            multilabel_count += 1
    print(multilabel_count) '''
    # paste here
    # dev_set_counter = 0  # to count to 1200. It is hard, I won't do developing.
    for file_name in file_names:
        with open(file_name, 'r', encoding="latin-1") as f:
            data = f.read()

        print(file_name)
        data = data.strip('<!DOCTYPE lewis SYSTEM "lewis.dtd">')
        while data:
            train_or_test = ""
            index_of_train_or_test = data.index("LEWISSPLIT=")
            index_of_train_or_test_end = data.index("CGISPLIT")
            which = data[index_of_train_or_test + 12:index_of_train_or_test_end - 2]

            # find if it belongs to train or test set.
            if which == "TEST":
                train_or_test = "test"
            elif which == "TRAIN":
                train_or_test = "train"
            else:
                train_or_test = "neither"

            index_of_id = data.index("NEWID=")
            index_of_date = data.index("DATE")
            new_id = int(data[index_of_id:index_of_date].strip('NEWID="').strip('">\n<'))

            # find the topic(s) of the article; if it is not in the max 10 topics, dont preprocess it.
            if new_id in id_list_training or new_id in id_list_test:
                if train_or_test == "train":
                    topics_current = sorted_id_topics_training_set[new_id]
                elif train_or_test == "test":
                    topics_current = sorted_id_topics_test_set[new_id]

                # If train, put in train tf.idf; if test, put in test tf.idf
                if train_or_test == "train" or train_or_test == "test":  # If neither, don't preprocess.
                    if "<TITLE>" not in data:
                        index_of_reuters_end = data.index('</REUTERS>')
                        # handled_article = data[:index_of_reuters_end]
                        data = data[index_of_reuters_end + 11:]  # I deleted the handled data by using slicing.
                        continue
                    else:
                        index_of_title = data.index("<TITLE>")
                        index_of_title_end = data.index("</TITLE>")

                    title = data[index_of_title + 7:index_of_title_end]
                    title = title.lower()  # case folding
                    title = punctuation_removal(title)  # punctuation removal for title
                    title = stop_word_removal(title)  # title is now a list, containing the words in the title

                    if "<BODY>" not in data:
                        index_of_reuters_end = data.index('</REUTERS>')
                        # handled_article = data[:index_of_reuters_end]
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

                    mini_vocab = list(indexes.keys())
                    vocabulary.update(set(title))  # for naive bayes vocabulary size |V|
                    # deneme olarak set title, instead of set mini_vocab yapiyorum.
                    # tuple = (new_id, indexes) is the element of the list of inverted_index[word]
                    set_of_words = set(title)
                    set_of_words = list(set_of_words)

                    # If train, put in train tf.idf; if test, put in test tf.idf
                    if train_or_test == "train":
                        for word in title:  # adding the id of the file to the postings of the words in body
                            check_list = []  # checking list
                            if word in inverted_index_train:
                                for i in range(len(inverted_index_train[word])):
                                    check_list.append(inverted_index_train[word][0])  # collect the doc_ids.

                                if new_id not in check_list:
                                    inverted_index_train[word].append(
                                        (new_id, indexes[word]))  # new_id, list of indexes as a tuple
                            else:
                                inverted_index_train[word] = [
                                    (new_id, indexes[word])]  # new_id, list of indexes as a tuple

                        for word in set_of_words:
                            term_freq = title.count(word)
                            if term_freq > 0:
                                term_freq = 1 + math.log(term_freq,
                                                         10)  # Put the TF into the dict, later calculate tf.idf

                            vector_space_train[new_id][word] = term_freq
                            # calculate tf.idf later because we cannot calculate idf now.

                            # for naive bayes part, filling the mega_doc for the topics with word counts.
                            # we need this part only for training of naive bayes.
                            for topic_ in topics_current:
                                if word in mega_doc[topic_]:
                                    mega_doc[topic_][word] = mega_doc[topic_][word] + len(indexes[word])
                                else:
                                    mega_doc[topic_][word] = len(indexes[word])
                    elif train_or_test == "test":
                        testing_dict[new_id] = title  # put the sentences in this dict, to test later.
                        for word in title:  # adding the id of the file to the postings of the words in body
                            check_list = []  # checking list
                            if word in inverted_index_test:
                                for i in range(len(inverted_index_test[word])):
                                    check_list.append(inverted_index_test[word][0])  # collect the doc_ids.

                                if new_id not in check_list:
                                    inverted_index_test[word].append(
                                        (new_id, indexes[word]))  # new_id, list of indexes as a tuple
                            else:
                                inverted_index_test[word] = [
                                    (new_id, indexes[word])]  # new_id, list of indexes as a tuple

                        for word in set_of_words:
                            term_freq = title.count(word)
                            if term_freq > 0:
                                term_freq = 1 + math.log(term_freq,
                                                         10)  # Put the TF into the dict, later calculate tf.idf

                            vector_space_test[new_id][word] = term_freq
                            # calculate tf.idf later because we cannot calculate idf now.

            # calculate tf.idf score for each unique word in the document and add it to the dictionary
            # vector_space[new_id][word] = tf.idf score value, then normalize.

            # print(inverted_index)
            index_of_reuters_end = data.index('</REUTERS>')
            # handled_article = data[:index_of_reuters_end]
            data = data[index_of_reuters_end + 11:]  # I deleted the handled data by using slicing.

    # Filling the mega_doc_count dict, which holds the n values.
    for topic_ in max_ten_topics:
        list_values = list(mega_doc[topic_].values())
        mega_doc_count[topic_] = sum(list_values)



    size_of_vocab = len(vocabulary)
    print("size of vocabulary : ", size_of_vocab)
    # Now we have mega_doc, mega_doc_count, vocabulary, so we can calculate P(w_k,c_j) parameter.
    # Also, we had topics_train_len that have the info for P(c_j) parameter. So we did the training part of MNBayes.

    # Calculating the tf.idf values and putting them into the vector_space
    # For train/test, put in train/test tf.idf, vice versa.
    for doc in range(1, 21579):
        sum_ = 0
        for word in vector_space_train[doc]:
            tf = vector_space_train[doc][word]
            df = len(inverted_index_train[word])  # not sure if it returns the document frequency
            x = 6494 / df
            idf = math.log(x, 10)
            tf_idf = tf * idf  # tf.idf calculated
            vector_space_train[doc][word] = tf_idf  # not normalized tf.idf, I want to make the vector's length = 1.
            sum_ += tf_idf ** 2  # square of the value is taken to normalize later.

        vector_length = math.sqrt(sum_)
        for word in vector_space_train[doc]:
            vector_space_train[doc][word] = vector_space_train[doc][word] / vector_length  # normalized value.
# For test
    for doc in range(1, 21579):
        sum_ = 0
        for word in vector_space_test[doc]:
            tf = vector_space_test[doc][word]
            df = len(inverted_index_test[word])  # not sure if it returns the document frequency
            x = 2548 / df
            idf = math.log(x, 10)
            tf_idf = tf * idf  # tf.idf calculated
            vector_space_test[doc][word] = tf_idf  # not normalized tf.idf, I want to make the vector's length = 1.
            sum_ += tf_idf ** 2  # square of the value is taken to normalize later.

        vector_length = math.sqrt(sum_)
        for word in vector_space_test[doc]:
            vector_space_test[doc][word] = vector_space_test[doc][word] / vector_length  # normalized value.

    # Naive Bayes Learning and Testing.
    P_c_j_dict = dict()
    number_of_docs = 6494  # for training.
    for topicAndValue in sorted_train:
        P_c_j_dict[topicAndValue[0]] = topicAndValue[1] / number_of_docs

    vocab = list(vocabulary)
    # calculate P(w_k|c_j) for each word w_k in vocab.
    P_w_k_c_j_dict = dict()
    for word in vocab:
        P_w_k_c_j_dict[word] = {}
        for topic in max_ten_topics:
            if word in mega_doc[topic]:
                P_w_k_c_j_dict[word][topic] = (mega_doc[topic][word] + 1)/(mega_doc_count[topic]+size_of_vocab)
            else:  # Put zero if not in mega_doc.
                P_w_k_c_j_dict[word][topic] = 1 / (mega_doc_count[topic] + size_of_vocab)
    # now we have probabilities and we have learned these parameters, we can use them.
    result_dict = dict()  # {id : {class:prob, class2:prob}, then we sort and look at the max probs.
    for testing_id in list(testing_dict.keys()):
        result_dict[testing_id] = {}

    for id, words in testing_dict.items():
        resulting_probabilities = dict()
        max_topic = ""
        for topic, P_c_j in P_c_j_dict.items():
            log_P_c_j = math.log(P_c_j,2)
            sum_log_Pxicj = 0
            for word in words:
                Pxicj = math.log(P_w_k_c_j_dict[word][topic],2)
                sum_log_Pxicj += Pxicj

            sum_for_argmax = log_P_c_j + sum_log_Pxicj
            result_dict[id][topic] = sum_for_argmax  # we tested this id for this topic, do this for 10 topics.

    # We filled the result_dict, now we can sort the values in the dict and look at the max probs.
    sorted_result_dict = dict()
    for id, probs_dict in result_dict.items():
        sorted_result_dict[id] = {}

    sorted_in_0_1 = {}
    answer_topics_dict = {}
    # Confusion matrix is a list: [tp, fp, fn, tn]
    # a dict for micro, macro avg. is created:
    micro_macro_dict = {}
    max_ten_topics = ['trade', 'money-fx', 'grain', 'crude', 'acq', 'interest', 'wheat', 'corn', 'ship', 'earn']
    for topic in max_ten_topics:
        micro_macro_dict[topic] = [0, 0, 0, 0]

    for id, probs_dict in result_dict.items():
        topics_current = sorted_id_topics_test_set[id]  # for testing predictions in the probs_dict.
        sorted_result_list = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        sum = 0
        for topicAndValue in sorted_result_list:
            sum += topicAndValue[1]

        for topicAndValue in sorted_result_list:
            sorted_in_0_1[topicAndValue[0]] = topicAndValue[1] / sum

        sorted_result_list = sorted(sorted_in_0_1.items(), key=lambda x: x[1])
        answer_topics_list = []
        for topicAndValue in sorted_result_list:
            if topicAndValue[1] / sorted_result_list[0][1] < 1.011:
                answer_topics_list.append(topicAndValue[0])

        answer_topics_dict[id] = answer_topics_list
        not_in_answer_topics_list = []
        for topic in max_ten_topics:
            if topic not in answer_topics_list:
                not_in_answer_topics_list.append(topic)

        for topic_ in answer_topics_list:
            if topic_ in sorted_id_topics_test_set[id]:
                micro_macro_dict[topic_][0] += 1  # tp increment
            else:
                micro_macro_dict[topic_][1] += 1  # fp increment

        for topic_ in not_in_answer_topics_list:
            if topic_ in sorted_id_topics_test_set[id]:
                micro_macro_dict[topic_][2] += 1  # fn increment
            else:
                micro_macro_dict[topic_][3] += 1  # tn increment

        # if int(id) > 20000:
        #   print(sorted_id_topics_test_set[id])
        #  print(sorted_result_list)

    micro_avg_precision = 0
    micro_avg_recall = 0
    micro_avg_F_score = 0
    macro_avg_precision = 0
    macro_avg_precision_denom = 0
    macro_avg_recall = 0
    macro_avg_recall_denom = 0
    macro_avg_F_score = 0

    # calculation:
    for topic, conf_matrix in micro_macro_dict.items():
        tp = conf_matrix[0]
        fp = conf_matrix[1]
        fn = conf_matrix[2]
        tn = conf_matrix[3]
        micro_avg_precision += tp / (tp + fp)
        micro_avg_recall += tp / (tp + fn)
        macro_avg_precision += tp
        macro_avg_precision_denom += tp + fp
        macro_avg_recall += tp
        macro_avg_recall_denom += tp + fn

    micro_avg_precision = micro_avg_precision / 10
    micro_avg_recall = micro_avg_recall / 10
    macro_avg_precision = macro_avg_precision / macro_avg_precision_denom
    macro_avg_recall = macro_avg_recall / macro_avg_recall_denom
    micro_avg_F_score = 2 * micro_avg_precision * micro_avg_recall / (micro_avg_precision + micro_avg_recall)
    macro_avg_F_score = 2 * macro_avg_precision * macro_avg_recall / (macro_avg_precision + macro_avg_recall)

    print("For Naive Bayes part evaluation: ")
    print("micro_avg_precision:", micro_avg_precision)
    print("micro_avg_recall:", micro_avg_recall)
    print("macro_avg_precision:", macro_avg_precision)
    print("macro_avg_recall:", macro_avg_recall)
    print("micro_avg_F_score:", micro_avg_F_score)
    print("macro_avg_F_score:", macro_avg_F_score)
    # Naive Bayes is over
    # Let's look at KNN:
    cosine_similarity_dict = {}
    for id_test in list(sorted_id_topics_test_set.keys()):
        cosine_similarity_dict[id_test] = {}  # I opened an empty dict as value for the key (id of test document)
        # This dict opened as value = {id_train1:score, id_train2: score, ....}
        # Then, I will sort these according to scores and find the K neighborhood of the test document.

    for id_test in list(sorted_id_topics_test_set.keys()):
        for id_train in list(sorted_id_topics_training_set.keys()):
            product = 0
            for word_train, tf_idf_of_word_train in vector_space_train[id_train].items():
                for word_test, tf_idf_of_word_test in vector_space_test[id_test].items():
                    if word_train == word_test:
                        product += tf_idf_of_word_train * tf_idf_of_word_test
            cosine_similarity_dict[id_test][id_train] = product

    # Sort the dict according to the cosine similarity, that is value(a list) of the dictionary key(test id).
    for id_test in list(cosine_similarity_dict.keys()):
        cosine_similarity_dict[id_test] = \
            {k: v for k, v in sorted(cosine_similarity_dict[id_test].items(), key=lambda item: item[1], reverse=True)}

    first_K_dict = {}
    for id_test in list(sorted_id_topics_test_set.keys()):
        K = 5  # the infamous K, it can be changed, however 5 is a great number.
        # first_K_dict[id_test] = {k: v for k, v in cosine_similarity_dict[id_test].items()[:K]}
        first_K_dict[id_test] = [k for k in list(cosine_similarity_dict[id_test].keys())[:K]]

    # Multilabelled answers are explained in the report, yet not coded.