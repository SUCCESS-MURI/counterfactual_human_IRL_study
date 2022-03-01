
    var html; 
  
    // setup stimulus
    var html = '<div>'
    html += '<video id="jspsych-video-keyboard-response-stimulus"';

    if(trial.width) {
      html += ' width="'+trial.width+'"';
    }
    if(trial.height) {
      html += ' height="'+trial.height+'"';
    }
    if(trial.autoplay){
      html += " autoplay ";
    }
    if(trial.controls){
      html +=" controls ";
    }
    html +=">";

    var video_preload_blob = jsPsych.pluginAPI.getVideoBuffer(trial.sources[0]);
    if(!video_preload_blob) {
      for(var i=0; i<trial.sources.length; i++){
        var file_name = trial.sources[i];
        if(file_name.indexOf('?') > -1){
          file_name = file_name.substring(0, file_name.indexOf('?'));
        }
        var type = file_name.substr(file_name.lastIndexOf('.') + 1);
        type = type.toLowerCase();
        html+='<source src="' + file_name + '" type="video/'+type+'">';   
      }
    }
    html += "</video>";
    html += "</div>";


if(trial.scale_width !== null){
  var w = trial.scale_width + 'px';
} else {
  var w = '100%';
}

// show preamble text
if(trial.preamble !== null){
  html += '<div id="jspsych-html-button-response-preamble" class="jspsych-html-button-response-preamble">'+trial.preamble+'</div>';
}


// display stimulus
html += '<div id="jspsych-html-button-response-stimulus">'+trial.stimulus+'</div>';



html += '<div class="jspsych-image-slider-response-container" style="position:relative; margin: 0 auto 3em auto; ';
if(trial.slider_width !== null){
html += 'width:'+trial.slider_width+'px;';
}
html += '">';
html += '<input type="range" value="'+trial.start+'" min="'+trial.min+'" max="'+trial.max+'" step="'+trial.step+'" style="width: 100%;" id="jspsych-image-slider-response-response"></input>';
html += '<div>'
for(var j=0; j < trial.labels.length; j++){
var width = 100/(trial.labels.length-1);
var left_offset = (j * (100 /(trial.labels.length - 1))) - (width/2);
html += '<div style="display: inline-block; position: absolute; left:'+left_offset+'%; text-align: center; width: '+width+'%;">';
html += '<span style="text-align: center; font-size: 80%;">'+trial.labels[j]+'</span>';
html += '</div>'
}
html += '</div>';
html += '</div>';
html += '</div>';

//show prompt if there is one
if (trial.prompt !== null) {
  html += trial.prompt;
}

html += '<style id="jspsych-survey-likert-css">';
html += ".jspsych-survey-likert-statement { display:block; font-size: 16px; padding-top: 40px; margin-bottom:10px; }"+
  ".jspsych-survey-likert-opts { list-style:none; width:"+w+"; margin:auto; padding:0 0 35px; display:block; font-size: 14px; line-height:1.1em; }"+
  ".jspsych-survey-likert-opt-label { line-height: 1.1em; color: #444; }"+
  ".jspsych-survey-likert-opts:before { content: ''; position:relative; top:11px; /*left:9.5%;*/ display:block; background-color:#efefef; height:4px; width:100%; }"+
  ".jspsych-survey-likert-opts:last-of-type { border-bottom: 0; }"+
  ".jspsych-survey-likert-opts li { display:inline-block; /*width:19%;*/ text-align:center; vertical-align: top; }"+
  ".jspsych-survey-likert-opts li input[type=radio] { display:block; position:relative; top:0; left:50%; margin-left:-6px; }"
html += '</style>';

// show preamble text

html += '<form id="jspsych-survey-likert-form">';

// add likert scale questions ///
// generate question order. this is randomized here as opposed to randomizing the order of trial.questions
// so that the data are always associated with the same question regardless of order
var question_order = [];
for(var i=0; i<trial.questions.length; i++){
  question_order.push(i);
}
if(trial.randomize_question_order){
  question_order = jsPsych.randomization.shuffle(question_order);
}

for (var i = 0; i < trial.questions.length; i++) {
  var question = trial.questions[question_order[i]];
  // add question
  html += '<label class="jspsych-survey-likert-statement">' + question.prompt + '</label>';
  // add options
  var width = 100 / question.labels.length;
  var options_string = '<ul class="jspsych-survey-likert-opts" data-name="'+question.name+'" data-radio-group="Q' + question_order[i] + '">';
  for (var j = 0; j < question.labels.length; j++) {
    options_string += '<li style="width:' + width + '%"><input type="radio" name="Q' + question_order[i] + '" value="' + j + '"';
    if(question.required){
      options_string += ' required';
    }
    options_string += '><label class="jspsych-survey-likert-opt-label">' + question.labels[j] + '</label></li>';
  }
  options_string += '</ul>';
  html += options_string;
}

// add submit button
html += '<input type="submit" id="jspsych-survey-text-next" class="jspsych-btn jspsych-survey-text" value="'+trial.button_label+'"></input>';

html += '</form>'
display_element.innerHTML = html;

// backup in case autofocus doesn't work
display_element.querySelector('#input-'+question_order[0]).focus();

display_element.querySelector('#jspsych-survey-likert-form').addEventListener('submit', function(e){
e.preventDefault();
// measure response time
var endTime = performance.now();
var response_time = endTime - startTime;

// create object to hold responses
var question_data = {};
var matches = display_element.querySelectorAll('#jspsych-survey-likert-form .jspsych-survey-likert-opts');
for(var index = 0; index < matches.length; index++){
var id = matches[index].dataset['radioGroup'];
var el = display_element.querySelector('input[name="' + id + '"]:checked');
if (el === null) {
  var response = "";
} else {
  var response = parseInt(el.value);
}
var obje = {};
if(matches[index].attributes['data-name'].value !== ''){
  var name = matches[index].attributes['data-name'].value;
} else {
  var name = id;
}
obje[name] = response;
Object.assign(question_data, obje);
}

// save data
var trial_data = {
"rt": response_time,
"responses": JSON.stringify(question_data),
"question_order": JSON.stringify(question_order)
};

display_element.innerHTML = '';

// next trial
jsPsych.finishTrial(trial_data);
});

var startTime = performance.now();
};

return plugin;
})();
