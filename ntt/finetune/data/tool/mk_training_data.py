import sys, os
import json
import re

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        #python mk_training_data.py ../../../../../../../../analysis/tool/dialogue/B
        sys.stderr.write('Usage: [Input dirname (jsonl)]\n')
        sys.exit()

    input_dirname = args[1]
    list_filenames = os.listdir(input_dirname)
    for filename in list_filenames:
        input_filename = input_dirname + '/' + filename
        input_file = open(input_filename)
        dialogue = []
        in_chat = False
        for line in input_file:
            line_buf = line.rstrip()
            data = json.loads(line_buf)
            speaker = data['speaker']
            text = data['text']
            if 'といえば' in text:
                in_chat = True
            elif 'そろそろ時間' in text:
                in_chat = False

            if in_chat:
                dialogue.append([speaker, text])
        input_file.close()


        if len(dialogue) == 0:
            continue
        
        context = ''
        response = ''
        if dialogue[-1][0] == 'system':
            response = dialogue.pop()[1]
        else:
            dialogue.pop()
            response = dialogue.pop()[1]
        
        for i, turn in enumerate(dialogue):
            if i == 0:
                m = re.match('あ、(.+)といえば', turn[1]) 
                if m:
                    context += '<|user|>%s<|endofuser|>' % (m.group(1))
            if turn[0] == 'user':
                context += '<|user|>%s<|endofuser|>' % (turn[1])
            elif turn[0] == 'system':
                context += '%s<|endofassistant|>' % (turn[1])

        out_data = {'prompt': context,
                    'response': response}
        print(json.dumps(out_data, ensure_ascii=False))
