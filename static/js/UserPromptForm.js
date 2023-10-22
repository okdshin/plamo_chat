export default {
    props: {},
    data() {
        return {
            query_and_reply_list: [],
            is_processing: false,
            input_text: '',
            current_processing_input_text: '',
            stored_texts: '',
        }
    },
    created : function(){
        /*
        let self = this;
        let es = new EventSource('/api/v1/generate_text/')
        console.log("here");
        es.onopen = () => {
            console.log('SSE connection opened');
        };
        es.onerror = (error) => {
            es.close();
            console.error('SSE connection error', error);
        };
        es.onmessage = (event) => {
            console.log('onmessage', event);
            let res = JSON.parse(event.data)
            if(res.text == null) {
                self.query_and_reply_list.push({
                    text: self.stored_texts
                });
                self.stored_texts = []
                self.is_processing = false;
            }
            else {
                self.stored_texts.push(res.text)
            }
            //this.id = res.id
        };
        */
    },
    methods: {
        submitUserPrompt() {
            let self = this;
            self.is_processing = true;
            let options = {
                input_text: self.input_text,
                stream: true,
                exclude_input: true,
            };
            self.current_processing_input_text = self.input_text;
            console.log(options);
            axios
                .post('/api/v1/generate_text/', options,
                    {
                        //responseType: 'stream',
                        onDownloadProgress: (progressEvent) => {
                            console.log(progressEvent);
                            const responseText = progressEvent.event.target.responseText;
                            const regex = /"text": "([^"]+)"/g;
                            const array = [...responseText.matchAll(regex)];
                            const generated_text = array.map((a) => a[1]).join('');
                            self.stored_texts = generated_text;
                        },
                    }
                )
                .then(async (response) => {
                    const responseText = response.data;
                    const regex = /"text": "([^"]+)"/g;
                    const array = [...responseText.matchAll(regex)];
                    const generated_text = array.map((a) => a[1]).join('');
                    console.log('Received', generated_text);
                    self.current_processing_input_text = "";
                    self.stored_texts = "";
                    self.query_and_reply_list.push({
                        prompt: options.input_text,
                        generated_text: generated_text,
                    });
                })
                .catch((error) => {
                    console.error('Error:', error);
                    self.stored_texts = "ERROR"
                })
                .finally(function() {
                    self.is_processing = false;
                });
        }
    },
    template: `
    <form>
        <div class="form-group">
            <textarea v-model="input_text" class="form-control" id="input_text" rows="3" placeholder="input text here"></textarea>
        </div>
        <button @click="submitUserPrompt" :disabled="is_processing || (input_text.length === 0)" type="button" class="mt-1 col-12 btn btn-primary">
            <template v-if="is_processing">
                <div class="spinner-border" role="status" style="width:1rem; height: 1rem"></div>
                Processing...
            </template>
            <template v-else>
                <i class="bi bi-send fs-8"></i>
                Send
            </template>
        </button>
    </form>
    <li class="list-group-item pt-2">
        <template v-if="stored_texts">
            <generation-log
                :prompt="current_processing_input_text"
                :generated_text="stored_texts"
            />
        </template>
        <template v-for="item in query_and_reply_list.slice().reverse()">
            <generation-log
                :prompt="item.prompt"
                :generated_text="item.generated_text"
            />
        </template>
    </li>
    `
}
