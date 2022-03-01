/**
 * jspsych-survey-text
 * a jspsych plugin for free response survey questions
 *
 * Josh de Leeuw
 *
 * documentation: docs.jspsych.org
 *
 */


jsPsych.plugins['survey-text'] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'survey-text',
    description: '',
    parameters: {
      questions: {
        type: jsPsych.plugins.parameterType.COMPLEX,
        array: true,
        pretty_name: 'Questions',
        default: null,
        nested: {
          prompt: {
            type: jsPsych.plugins.parameterType.HTML_STRING,
            pretty_name: 'Prompt',
            default: null,
            description: 'Prompt for the subject to response'
          },
          placeholder: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Value',
            default: "",
            description: 'Placeholder text in the textfield.'
          },
          rows: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Rows',
            default: 1,
            description: 'The number of rows for the response text box.'
          },
          columns: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Columns',
            default: 40,
            description: 'The number of columns for the response text box.'
          },
          required: {
            type: jsPsych.plugins.parameterType.BOOL,
            pretty_name: 'Required',
            default: false,
            description: 'Require a response'
          },
          name: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Question Name',
            default: '',
            description: 'Controls the name of data values associated with this question'
          },
          qtype: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Question Type',
            default: '',
            description: 'What type of question is being posed?'
          }
        }
      },
      preamble: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Preamble',
        default: null,
        description: 'HTML formatted string to display at the top of the page above all the questions.'
      },
      button_label: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        description: 'The text that appears on the button to finish the trial.'
      },
      // MISC
      domain: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Domain',
        default: null,
        description: 'Identifies which domain the primary question was in.'
      },
      condition: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Condition',
        default: null,
        description: 'Identifies which condition the participant was assigned to.'
      },
      question_type: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Question Type',
        default: null,
        description: 'Identifies what type of question was asked.'
      },
        randomize_question_order: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Randomize question order',
        default: null,
        description: 'Randomize question order.'
      },
      counterbalance: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Counterbalance',
        default: null,
        description: 'Identifies which counterbalancing the participant was assigned to.'
      },
      train_test_set: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Train test set',
        default: null,
        description: 'Identifies which train/test set the participant was assigned to.'
      }
    }
  }

  plugin.trial = function(display_element, trial) {

    for (var i = 0; i < trial.questions.length; i++) {
      if (typeof trial.questions[i].rows == 'undefined') {
        trial.questions[i].rows = 1;
      }
    }
    for (var i = 0; i < trial.questions.length; i++) {
      if (typeof trial.questions[i].columns == 'undefined') {
        trial.questions[i].columns = 40;
      }
    }
    for (var i = 0; i < trial.questions.length; i++) {
      if (typeof trial.questions[i].value == 'undefined') {
        trial.questions[i].value = "";
      }
    }

    var html = '';
    // show preamble text
    if(trial.preamble !== null){
      html += '<div id="jspsych-survey-text-preamble" class="jspsych-survey-text-preamble">'+trial.preamble+'</div>';
    }
    // start form
    html += '<form id="jspsych-survey-text-form">'

    // generate question order
    var question_order = [];
    for(var i=0; i<trial.questions.length; i++){
      question_order.push(i);
    }
    if(trial.randomize_question_order){
      question_order = jsPsych.randomization.shuffle(question_order);
    }

  // add questions
  for (var i = 0; i < trial.questions.length; i++) {
      // create questiion container
      var question = trial.questions[question_order[i]];
      var question_id = question_order[i];

      if (question.qtype == "multi") {
        var autofocus = i == 0 ? "autofocus" : "";
        html += '<div id="jspsych-survey-multi-choice-'+question_id+'" data-name="'+question.name+' ' + autofocus +'">';
        // add question text
        html += '<br>'
        html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><b>' + question.prompt
        if(question.required){
          html += "<span class='required'>*</span>";
        }
        html += '</b></p>';

        for(var j = 0; j < question.options.length; j++) {
          var option_id_name = "jspsych-survey-multi-choice-option-"+question_id+"-"+j;
          var input_name = question_id;
          var input_id = 'jspsych-survey-multi-choice-response-'+question_id+'-'+j;
          var required_attr = question.required ? 'required' : '';
          html += '<input type="radio" name="'+input_name+'" id="' + i + '-'+ j + '" style="margin-left:15px" value="'+question.options[j]+'"' + required_attr +'> ' + question.options[j] + '</input>';
        }
        html += '</div>';

      // text and numeric have the same setup
      } else {
        html += '<div id="jspsych-survey-text-'+question_id+'" class="jspsych-survey-text-question" style="margin: 2em 0em;">';
        html += '<p class="jspsych-survey-text">' + question.prompt + '</p>';
        var autofocus = i == 0 ? "autofocus" : "";
        var req = question.required ? "required" : "";
        if(question.rows == 1){
          html += '<input type="' + question.qtype + '" id="input-'+question_id+'"  name="#jspsych-survey-text-response-' + question_id + '" data-name="'+question.name+'" size="'+question.columns+'" '+autofocus+' '+req+' placeholder="'+question.placeholder+'"></input>';
        } else {
          html += '<textarea id="input-'+question_id+'" name="#jspsych-survey-text-response-' + question_id + '" data-name="'+question.name+'" cols="' + question.columns + '" rows="' + question.rows + '" '+autofocus+' '+req+' placeholder="'+question.placeholder+'"></textarea>';
        }
        html += '</div>';
      }
    }

    // // backup in case autofocus doesn't work
    // display_element.querySelector('#input-'+question_order[0]).focus();

    // add submit button
    html += '<br><br><input type="submit" id="jspsych-survey-text-next" class="jspsych-btn jspsych-survey-text" value="'+trial.button_label+'"></input>';

    html += '</form>'
    display_element.innerHTML = html;

    // // backup in case autofocus doesn't work
    // display_element.querySelector('#input-'+question_order[0]).focus();

    var curr_options = [-1, -1, -1, -1]

    function check_id_at_name(name) {
      id = $('input[name=' + name + ']:checked').attr('id')
      id = id.substr(id.length-1)

      // disable this option for any other preference #, since people should only choose something once
      for (var i = 0; i < curr_options.length; i++) {
        if (i != name) {
          document.getElementById(i + '-' + id).setAttribute('disabled', 'disabled')
        }
      }

      if (curr_options[name] == -1) {
        curr_options[name] = id
      } else {
        for (var i = 0; i < curr_options.length; i++) {
          document.getElementById(i + '-' + curr_options[name]).removeAttribute('disabled')
        }
        curr_options[name] = id
      }
    }

    display_element.querySelector('#jspsych-survey-text-form').addEventListener('submit', function(e) {
      e.preventDefault();
      // measure response time
      var endTime = performance.now();
      var response_time = endTime - startTime;

      // create object to hold responses
      var question_data = {};

      for(var index=0; index < trial.questions.length; index++){
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
        var trialdata = {
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
            "domain": null,
            "condition": trial.condition,
            "counterbalance": trial.counterbalance,
            "train_test_set": trial.train_test_set,
            "question_type": trial.question_type
        };

      display_element.innerHTML = '';


      // next trial
      jsPsych.finishTrial(trialdata);
    });

    var startTime = performance.now();
  };

  return plugin;
})();
