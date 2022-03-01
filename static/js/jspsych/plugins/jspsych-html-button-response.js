/**
 * jspsych-html-button-response
 * Josh de Leeuw
 *
 * plugin for displaying a stimulus and getting a keyboard response
 *
 * documentation: docs.jspsych.org
 *
 **/

// for iframe children
var path = null;

jsPsych.plugins["html-button-response"] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'html-button-response',
    description: '',
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: null,
        description: 'The HTML string to be displayed'
      },
      choices: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Choices',
        default: null,
        array: true,
        description: 'The labels for the buttons.'
      },
      button_html: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button HTML',
        default: '<button class="jspsych-btn">%choice%</button>',
        array: true,
        description: 'The html of the button. Can create own style.'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed under the button.'
      },
      practice_header: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Practice Header',
        default: null,
        description: 'Any content here will be displayed above the practice environment.'
      },
      practice_caption: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Practice Caption',
        default: null,
        description: 'Caption to use in Practice Environment'
      },
      stimulus_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show the trial.'
      },
      margin_vertical: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Margin vertical',
        default: '0px',
        description: 'The vertical margin of the button.'
      },
      margin_horizontal: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Margin horizontal',
        default: '8px',
        description: 'The horizontal margin of the button.'
      },
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Response ends trial',
        default: true,
        description: 'If true, then trial will end when user responds.'
      },
      showButtons: {
        type:jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'showButtons',
        default: true,
        description: 'Whether or not the buttons show'
      },
      // MISC
      question_type: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Question Type',
        default: null,
        description: 'Identifies what type of question was asked.'
      },
      textinput: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Text input',
        default: null,
        description: 'Placeholder for textbox'
      },
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
      iframe_src: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'iFrame src',
        default: null,
        description: 'The path to any content to be loaded into the iFrame'
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
        },
    }
  }

  plugin.trial = function(display_element, trial) {
    if (trial.iframe_src != null) {
      path = trial.iframe_src
    }

    if(trial.practice_header !== null) {
      var practice_env = true
    }

    var html = ''

    if (trial.question_type == "post_main_game") {
      var score = 0

      if (sessionStorage.getItem(trial.domain + '_low_a') == 'true') score += 1;
      if (sessionStorage.getItem(trial.domain + '_low_b') == 'true') score += 1;
      if (sessionStorage.getItem(trial.domain + '_medium_a') == 'true') score += 1;
      if (sessionStorage.getItem(trial.domain + '_medium_b') == 'true') score += 1;
      if (sessionStorage.getItem(trial.domain + '_high_a') == 'true') score += 1;
      if (sessionStorage.getItem(trial.domain + '_high_b') == 'true') score += 1;

      trial.stimulus = trial.stimulus.replace("TEMP_SCORE", score.toString())
    }

    // display stimulus
    if(trial.practice_header !== null){
      html += '<div id="jspsych-html-button-response-preamble" class="jspsych-html-button-response-preamble">'+trial.practice_header+'</div>';
    }
    html += '<div id="jspsych-html-button-response-stimulus">'+trial.stimulus+'</div>';
    if (trial.practice_caption !== null){
      html += '<p>' + trial.practice_caption + '</p>'
    }
    //show prompt if there is one
    if (trial.prompt !== null) {
      html += trial.prompt;
    }

    var key_fn = function(e) {
        //  if (e.shiftKey || e.ctrlKey || e.altKey) {
          // e.preventDefault();
        //} else {
          var key = e.keyCode;
          if (!((key == 8) || (key == 32) || (key == 46) || (key >= 35 && key <= 40) || (key >= 65 && key <= 90))) {
          e.preventDefault();
      //  }
      }
    }

    if(trial.textinput !== null){
        html += '<div id="jspsych-survey-text" class="jspsych-survey-text-question" style="margin: 2em 0em;">';
        html += '<input type="text" id="textarea" name="#jspsych-survey-text-response" size="40" required placeholder="'+trial.textinput+'"></input>';
        html += '</div>';
    }

    if(trial.showButtons){
    //display buttons
    var buttons = [];
    if (Array.isArray(trial.button_html)) {
      if (trial.button_html.length == trial.choices.length) {
        buttons = trial.button_html;
      } else {
        console.error('Error in html-button-response plugin. The length of the button_html array does not equal the length of the choices array');
      }
    } else {
      for (var i = 0; i < trial.choices.length; i++) {
        buttons.push(trial.button_html);
      }
    }
    html += '<div id="jspsych-html-button-response-btngroup">';
    for (var i = 0; i < trial.choices.length; i++) {
      var str = buttons[i].replace(/%choice%/g, trial.choices[i]);
      html += '<div class="jspsych-html-button-response-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-button-response-button-' + i +'" data-choice="'+i+'">'+str+'</div>';
    }
    html += '</div>';
  }

    display_element.innerHTML = html;

    if(trial.textinput !== null){
      document.getElementById('textarea').addEventListener('keydown', key_fn);
    }

    // start time
    var start_time = performance.now();

    // add event listeners to buttons
    for (var i = 0; i < trial.choices.length; i++) {
      display_element.querySelector('#jspsych-html-button-response-button-' + i).addEventListener('click', function(e){
        var choice = e.currentTarget.getAttribute('data-choice'); // don't use dataset for jsdom compatibility
        after_response(choice);
      });
    }

    // store response
    var response = {
      rt: null,
      button: null
    };

    // function to handle responses by the subject
    function after_response(choice) {

      // measure rt
      var end_time = performance.now();
      var rt = end_time - start_time;
      response.button = choice;
      response.rt = rt;

      // after a valid response, the stimulus will have the CSS class 'responded'
      // which can be used to provide visual feedback that a response was recorded
      display_element.querySelector('#jspsych-html-button-response-stimulus').className += ' responded';

      // disable all the buttons after a response
      var btns = document.querySelectorAll('.jspsych-html-button-response-button button');
      for(var i=0; i<btns.length; i++){
        //btns[i].removeEventListener('click');
        btns[i].setAttribute('disabled', 'disabled');
      }

      if (trial.response_ends_trial) {
        end_trial();
      }
    };

    // function to end trial when it is time
    function end_trial() {

      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      // gather the data to store for the trial
        var trial_data = {
            // Primary information
            "primary_stimulus": null,
            "primary_rt": performance.now() - start_time,
            "primary_answer": null,

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

      // clear the display
      display_element.innerHTML = '';

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };

    // hide image if timing is set
    if (trial.stimulus_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        display_element.querySelector('#jspsych-html-button-response-stimulus').style.visibility = 'hidden';
      }, trial.stimulus_duration);
    }

    // end trial if time limit is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        end_trial();
      }, trial.trial_duration);
    }

  };

  return plugin;
})();