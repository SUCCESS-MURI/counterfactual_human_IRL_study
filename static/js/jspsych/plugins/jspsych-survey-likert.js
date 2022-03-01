/**
 * jspsych-survey-likert
 * a jspsych plugin for measuring items on a likert scale
 *
 * Josh de Leeuw
 *
 * documentation: docs.jspsych.org
 *
 */

jsPsych.plugins['survey-likert'] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'survey-likert',
    description: '',
    parameters: {
      questions: {
        type: jsPsych.plugins.parameterType.COMPLEX,
        array: true,
        pretty_name: 'Questions',
        nested: {

          prompt: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Prompt',
            default: undefined,
            description: 'Questions that are associated with the slider.'
          },
          labels: {
            type: jsPsych.plugins.parameterType.STRING,
            array: true,
            pretty_name: 'Labels',
            default: undefined,
            description: 'Labels to display for individual question.'
          },
          required: {
            type: jsPsych.plugins.parameterType.BOOL,
            pretty_name: 'Required',
            default: false,
            description: 'Makes answering the question required.'
          },
          name: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Question Name',
            default: '',
            description: 'Controls the name of data values associated with this question'
          }
        }
      },
      randomize_question_order: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Randomize Question Order',
        default: false,
        description: 'If true, the order of the questions will be randomized'
      },
      preamble: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Preamble',
        default: null,
        description: 'String to display at top of the page.'
      },
      scale_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Scale width',
        default: null,
        description: 'Width of the likert scales in pixels.'
      },
      button_label: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        description: 'Label of the button.'
      },

        // MISC
      condition: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Condition',
        default: null,
        description: 'Identifies which condition the participant was assigned to.'
      },
          domain: {
          type: jsPsych.plugins.parameterType.STRING,
          pretty_name: 'Domain',
          default: null,
          description: 'Identifies which domain the primary question was in.'
        },
        question_type: {
          type: jsPsych.plugins.parameterType.STRING,
          pretty_name: 'QuestionType',
          default: null,
          description: 'Identifies what type of question is being asked.'
        },
        attention_check_intended: {
          type: jsPsych.plugins.parameterType.STRING,
          pretty_name: 'IntendedResponse',
          default: null,
          description: 'Identifies what the appropriate answer to the attention check is.'
        }
    }
  }

  plugin.trial = function(display_element, trial) {

    var attn_checks = []

    if(trial.scale_width !== null){
      var w = trial.scale_width + 'px';
    } else {
      var w = '100%';
    }

    var html = "";
    // inject CSS for trial
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
    if(trial.preamble !== null){
      html += '<div id="jspsych-survey-likert-preamble" class="jspsych-survey-likert-preamble">'+trial.preamble+'</div>';
    }
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




    // mike added this the code below to support free form text responses in addition to likert scale surveys
    // todo: ensure that the data is being saved correctly
      // generate question order
    var freeform_order = [];
    for(var i=0; i<trial.freeform.length; i++){
      freeform_order.push(i);
    }

      if(trial.randomize_question_order){
      freeform_order = jsPsych.randomization.shuffle(freeform_order);
    }

    for (var i = 0; i < trial.freeform.length; i++) {

        var freeform = trial.freeform[freeform_order[i]];
        var freeform_id = freeform_order[i];


        html += '<div id="jspsych-survey-text-' + freeform_id + '" class="jspsych-survey-text-question" style="margin: 2em 0em;">';
        html += '<p class="jspsych-survey-text">' + freeform.prompt + '</p>';
        var autofocus = i == 0 ? "autofocus" : "";
        var req = freeform.required ? "required" : "";
        if (freeform.rows == 1) {
            html += '<input type="' + freeform.qtype + '" id="input-' + freeform_id + '"  name="#jspsych-survey-text-response-' + freeform_id + '" data-name="' + freeform.name + '" size="' + freeform.columns + '" ' + autofocus + ' ' + req + ' placeholder="' + freeform.placeholder + '"></input>';
        } else {
            html += '<textarea id="input-' + freeform_id + '" name="#jspsych-survey-text-response-' + freeform_id + '" data-name="' + freeform.name + '" cols="' + freeform.columns + '" rows="' + freeform.rows + '" ' + autofocus + ' ' + req + ' placeholder="' + freeform.placeholder + '"></textarea>';
        }
        html += '</div>';
    }




    // add submit button
    html += '<input type="submit" id="jspsych-survey-likert-next" class="jspsych-survey-likert jspsych-btn" value="'+trial.button_label+'"></input>';

    html += '</form>'

    display_element.innerHTML = html;

    display_element.querySelector('#jspsych-survey-likert-form').addEventListener('submit', function(e){
      e.preventDefault();
      // measure response time
      var endTime = performance.now();
      var response_time = endTime - startTime;

      // create object to hold responses
      var question_data = {};

      // go through the Likert questions and save data
      var matches = display_element.querySelectorAll('#jspsych-survey-likert-form .jspsych-survey-likert-opts');
      for(var index = 0; index < matches.length; index++){
        var id = matches[index].dataset['radioGroup'];
        var el = display_element.querySelector('input[name="' + id + '"]:checked');

        if (trial.questions[index].attention_check_intended != null) { // TODO(pkoppol): attention check is always the last question in the survey
          if (parseInt(el.value) != trial.questions[index].attention_check_intended) {
            attn_checks.push(false)
          }
        }
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

        // go through the freeform text questions and save data
        for(var index=0; index < trial.freeform.length; index++){
            var id = "Q" + index;
            var q = document.querySelector('#jspsych-survey-text-' + index)
            if (q == null) {
                var el = display_element.querySelector('input[name="' + index + '"]:checked');
                var val = el.id.substr(el.id.length-1)
                var name = id

            } else {
                var q_element = q.querySelector('textarea, input');
                var val = q_element.value;
                var name = q_element.attributes['data-name'].value;
                if(name == ''){
                    name = id;
                }
            }
            var obje = {};
            obje[name] = val;
            Object.assign(question_data, obje);
        }


        // save data
        var trial_data = {
            // Primary information
            "primary_stimulus": JSON.stringify(question_order),
            "primary_rt": response_time,
            "primary_answer": JSON.stringify(question_data),

            // Testing information
            "test_mdp": null,

            // Likert information
            "likert_rt": null,
            "likert_response": null,
            "likert_question_order": null,

            // Study information
            "domain": trial.domain,
            "condition": trial.condition,
            "counterbalance": trial.counterbalance,
            "train_test_set": trial.train_test_set,
            "question_type": trial.question_type
        };

      display_element.innerHTML = '';

      // next trial
      jsPsych.finishTrial(trial_data);
    });

    var startTime = performance.now();
  };

  return plugin;
})();
