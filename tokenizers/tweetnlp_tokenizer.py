import copy
from .tokenizer import Tokenizer, Tokens
from . import DEFAULTS
import pexpect
import re

# DEFAULTS = {
#     'corenlp_classpath': os.getenv('CLASSPATH'),
#     'tweetnlp_classpath': '../../ark-tweet-nlp-0.3.2',
# }


class TweetNLPTokenizer(Tokenizer):
    '''
    This wraps the ark tweet nlp tokenizer and pos tagger.
    Does not output the original text.
    '''

    def __init__(self, **kwargs):
        """
                Args:
                    annotators: set that can include pos, lemma, and ner.
                    classpath: Path to the tweetnlp directory of jars
                    mem: Java heap memory
                """
        self.classpath = (kwargs.get('classpath') or
                          DEFAULTS['tweetnlp_classpath'])
        # default is '../ark-tweet-nlp-0.3.2'
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.mem = kwargs.get('mem', '2g')
        self.thread = kwargs.get('threads', '2')
        self.norm_dict = kwargs.get('norm_dict') #pass the dictionary as argument
        self._launch()



    def _launch(self):
        '''
        start the java program
        :return:
        '''
        annotators = ['tokenize', 'ssplit']
        if 'pos' in self.annotators:
            annotators.append('pos')
        cmd = ['java', '-XX:ParallelGCThreads=' + self.thread,
               '-Xmx' + self.mem,
               '-jar ' + self.classpath + '/ark-tweet-nlp-0.3.2.jar']
        # We use pexpect to keep the subprocess alive and feed it commands.
        # Because we don't want to get hit by the max terminal buffer size,
        # we turn off canonical input processing to have unlimited bytes.
        # spawn a bash shell and run the java command
        self.tweetnlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.tweetnlp.setecho(False)
        self.tweetnlp.sendline('stty -icanon')
        self.tweetnlp.sendline(' '.join(cmd))
        self.tweetnlp.delaybeforesend = 0
        self.tweetnlp.delayafterread = 0
        self.tweetnlp.expect_exact('Listening on stdin for input.  (-h for help)', searchwindowsize=100)
        #make a small test
        self.tweetnlp.sendline('This is test tweet.')
        self.tweetnlp.expect_exact('Detected text input format\r\n')
        self.tweetnlp.expect(r'.*\r\n')
        self.tweetnlp.buffer = b'' #ignore the test output

    def _convert(self,token):
        '''
        map some tokens to a regular form

        :return:
        '''
        #TODO:normalize
        if token =='???':
            return '<UKN>'
        if token in self.norm_dict:
            return self.norm_dict[token]

        return token
    @staticmethod
    def _filter(word):
        # filter urls and RT, do not add to word list
        URL_PATTERN = re.compile(r'http(s)?://\w+\.\w+(/\w+)*')
        # the complex URL_PATTERN takes too long to run XD
        # URL_PATTERN = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
        RESERVED_WORDS_PATTERN = re.compile(r'^(RT|FAV)')
        if re.match(URL_PATTERN,word) or re.match(RESERVED_WORDS_PATTERN,word):
            return None
        else:
            return word

    def tokenize(self, text):
        clean_text = text.replace('\n', ' ')

        self.tweetnlp.sendline(clean_text.encode('ascii',errors='ignore'))
        self.tweetnlp.expect_exact(clean_text.encode('ascii',errors='ignore').decode()+'\r\n')
        # Skip to start of output (may have been stderr logging messages)
        output = self.tweetnlp.before

        # strip the \r\n at the beginning
        fields = output.decode('utf-8').strip('\r\n\t ').split('\t')

        # \t separated fields
        # tokenized_text pos_tags confidence
        data = []
        try:
            assert(len(fields)==3)
        except AssertionError:
            print (fields)

        wordlist = [t for t in fields[0].split(' ') if self._filter(t)]
        poslist = [p for p in fields[1].split(' ')]
        token_dict = {
            'characterOffsetBegin': 0,
            'characterOffsetEnd': 0,
            'word': '',
        }
        if 'pos' in self.annotators:
            token_dict['pos'] = ''

        for i in range(len(wordlist)):
            token_dict['word'] = wordlist[i]
            if 'pos' in token_dict: token_dict['pos'] = poslist[i]
            if i == 0:
                token_dict['characterOffsetBegin'] = 0
            else:
                token_dict['characterOffsetBegin'] = prev_end

            prev_end = token_dict['characterOffsetEnd'] = token_dict['characterOffsetBegin'] + len(token_dict['word'])

            data.append((
                self._convert(token_dict['word']),
                None,  # the original text is not provided
                (token_dict['characterOffsetBegin'],
                 token_dict['characterOffsetEnd']),
                token_dict.get('pos', None),
                None, #no lemma
                None
            ))
        return Tokens(data, self.annotators)

    def shutdown(self):
        self.tweetnlp.close()
        return


#testtok = TweetNLPTokenizer(annotators=['pos'])
#testtok.tokenize('Profit-Taking Hits Nikkei http://t.co/hVWpiDQ1 http://t.co/xJSPwE2z RT @WSJmarkets')
