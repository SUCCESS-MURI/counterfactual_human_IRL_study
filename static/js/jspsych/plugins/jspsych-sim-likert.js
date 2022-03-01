/**
 * jspsych-html-button-response
 * Josh de Leeuw
 *
 * plugin for displaying a stimulus and getting a keyboard response
 *
 * documentation: docs.jspsych.org
 *
 **/

// variables necessary for iFrame
var path = null;
var mdp_parameters = null;
var primary_misc = {};

jsPsych.plugins["sim-likert"] = (function() {

  var plugin = {};

  jsPsych.pluginAPI.registerPreload('video-keyboard-response', 'stimulus', 'video');


  plugin.info = {
    name: 'sim-likert',
    description: '',
    parameters: {


      //PRIMARY QUESTION STIMULUS + QUESTION
      stimulus: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: null,
        description: 'The HTML string to be displayed'
      },
      stimulus_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      iframe_id: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'iFrame ID',
        default: null,
        description: 'The id of an iFrame (if applicable) to allow access to variables'
      },
      iframe_src: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'iFrame src',
        default: null,
        description: 'The path to any content to be loaded into the iFrame'
      },
      preamble: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Preamble',
        default: null,
        description: 'String to display at top of the page.'
      },
      legend: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Legend',
        default: null,
        description: 'String to display at bottom of the page.'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed under the button.'
      },

      caption: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Caption',
        default: null,
        description: 'Caption for Images in Domain 1, Rating Tasks'
      },

      //BUTTON RESPONSE
      showButtons: {
        type:jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'showButtons',
        default: true,
        description: 'Whether or not the buttons show'
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

      //LIKERT SCALE
    questions: {
        type: jsPsych.plugins.parameterType.COMPLEX,
        array: true,
        pretty_name: 'Questions',
        nested: {

          prompt: {
            type: jsPsych.plugins.parameterType.STRING,
            pretty_name: 'Prompt',
            default: null,
            description: 'Questions that are associated with the slider.'
          },
          labels: {
            type: jsPsych.plugins.parameterType.STRING,
            array: true,
            pretty_name: 'Labels',
            default: null,
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
      scale_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Scale width',
        default: null,
        description: 'Width of the likert scales in pixels.'
      },


      //TRIAL PARAMETERS
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show the trial.'
      },
      button_label: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        description: 'Label of the button.'
      },
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Response ends trial',
        default: true,
        description: 'If true, then trial will end when user responds.'
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

      textinput: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Text input',
        default: null,
        description: 'Placeholder for textbox'
      },
      question_type: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'QuestionType',
        default: null,
        description: 'Identifies what type of question is being asked.'
      },
      question_id: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Question Id',
        default: null,
        description: 'ID of the question being asked.'
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
      training_traj_length: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Training trajectory length',
        default: null,
        description: 'Identifies length of training trajectory.'
      },
      mdp_parameters: {
        type: jsPsych.plugins.parameterType.OBJECT,
        pretty_name: 'MDP parameters',
        default: {},
        description: 'MDP parameters of the testing simulation.'
      },
      continue_condition: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Continue condition',
        default: null,
        description: 'Requirements needed to continue on.'
      },
      randomize_question_order: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Randomize question order',
        default: null,
        description: 'Randomize question order.'
      },
    }
  }

  plugin.trial = function(display_element, trial) {
      mdp_parameters = trial.mdp_parameters;
      continue_condition = trial.continue_condition;
    if (trial.iframe_src != null) {
      path = trial.iframe_src
    }
    primary_misc = {};

      // setup stimulus
      var html = '';
      html += '<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">';
      html += '<link rel=\"stylesheet\" href=\"/static/css/style.css\" type=\"text/css\"/>';

      html += '<div>';


    /* START PRIMARY QUESTION */
    // show preamble text
    if(trial.preamble !== null){
      html += '<div id="jspsych-html-button-response-preamble" class="jspsych-html-button-response-preamble">'+trial.preamble+'</div>';
    }


      // display stimulus
      if (trial.legend != null) {
          html += '<div class="row"><div class="column">';
          html += '<div id="jspsych-html-button-response-stimulus">' + trial.stimulus + '</div>';
          if (trial.caption !== null) {
              html += '<p>' + trial.caption + '</p>'
          }
          html += '</div>';
          html += '<div class="column">';

          if (trial.legend !== null) {
              html += '<div id="jspsych-html-button-response-legend" class="jspsych-html-button-response-legend">' + trial.legend + '</div>';
          }
          html += '</div></div>';
      }
      else {
          html += '<div id="jspsych-html-button-response-stimulus">'+trial.stimulus+'</div>';
          if (trial.caption !== null){
      html += '<p>' + trial.caption + '</p>'
          }
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
        html += '<input type="text" id="textarea" pattern="[a-zA-Z]*" name="#jspsych-survey-text-response" size="40" required placeholder="'+trial.textinput+'"></input>';
        html += '</div>';
    }


    //show prompt if there is one
    if (trial.prompt !== null) {
      html += trial.prompt;
    }

    /*
    // add a legend if there is one
      if(trial.legend !== null){
      html += '<div id="jspsych-html-button-response-legend" class="jspsych-html-button-response-legend">'+trial.legend+'</div>';
    } */

    //display_element.innerHTML = html;

    /* END PRIMARY QUESTION */

    /* START CONFIDENCE SCALE */

    if(trial.scale_width !== null){
      var w = trial.scale_width + 'px';
    } else {
      var w = '100%';
    }

    html += '<style id="jspsych-survey-likert-css">';
    html += ".jspsych-survey-likert-statement { display:block; font-size: 16px; padding-top: 0px; margin-bottom:10px; }"+
      ".jspsych-survey-likert-opts { list-style:none; width:"+w+"; margin:auto; padding:0 0 35px; display:block; font-size: 14px; line-height:1.1em; }"+
      ".jspsych-survey-likert-opt-label { line-height: 1.1em; color: #444; }"+
      ".jspsych-survey-likert-opts:before { content: ''; position:relative; top:11px; /*left:9.5%;*/ display:block; background-color:#efefef; height:4px; width:100%; }"+
      ".jspsych-survey-likert-opts:last-of-type { border-bottom: 0; }"+
      ".jspsych-survey-likert-opts li { display:inline-block; /*width:19%;*/ text-align:center; vertical-align: top; }"+
      ".jspsych-survey-likert-opts li input[type=radio] { display:block; position:relative; top:0; left:50%; margin-left:-6px; }"
    html += '</style>';


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
     html += '<input type="submit" disabled id="jspsych-survey-likert-next" class="jspsych-survey-likert jspsych-btn" value="'+trial.button_label+'"></input>';

     html += '</form>'
     display_element.innerHTML = html;

    if(trial.textinput !== null){
      document.getElementById('textarea').addEventListener('keydown', key_fn);
    }
      /* END CONFIDENCE SCALE */



      /* BUTTON HANDLERS + DATA SAVING */

      // start and end times
      var primaryStartTime = performance.now();
      console.log("primaryStartTime recorded");
      var primaryEndTime = performance.now();

    display_element.querySelector('#jspsych-survey-likert-form').addEventListener('submit', function(e){
      e.preventDefault();
      // measure response time
      var likertEndTime = performance.now();
      console.log("likertEndTime recorded");
      var likert_rt = likertEndTime - primaryEndTime;
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

      if (trial.iframe_id != null) {
        var iframe = document.getElementById(trial.iframe_id)
        primary_response.button = iframe.contentWindow['final_data']
      }

      if (trial.textinput != null) {
        primary_response.button = document.getElementById("textarea").value
      }

        //todo: what does this do? I think this can count the number of times you've watched the video
        if (trial.question_type == 'training_video') {
            v = document.getElementById('stimulus')
            primary_misc['stimulus_video_loops'] = stimulus_video_loops;
            primary_misc['stimulus_vid_curr_time'] = v.currentTime;
            primary_misc['video_length'] = v.duration
        }

        // save data
        trial_data = {
            // Primary information
            "primary_stimulus": trial.stimulus,
            "primary_rt": primary_response.primary_rt,
            "primary_answer": primary_response.button,

            // Testing information
            "test_mdp": trial.mdp_parameters,

            // Likert information
            "likert_rt": likert_rt,
            "likert_response": JSON.stringify(question_data),
            "likert_question_order": JSON.stringify(question_order),

            // Study information
            "domain": trial.domain,
            "condition": trial.condition,
            "counterbalance": trial.counterbalance,
            "train_test_set": trial.train_test_set,
            "question_type": trial.question_type
        };

      // remove listeners
      if(trial.textinput !== null){
        document.getElementById('textarea').removeEventListener('keydown', key_fn);
      }
      display_element.innerHTML = '';

      // next trial
      jsPsych.finishTrial(trial_data);
    });

      // store response
      var primary_response = {
        primary_rt: null,
        button: null
      };

      var secondary_response = {
        secondary_rt: null,
        key_press: null
      }

      function set_rt_and_enable_submit() {
          if(trial.iframe_id !== null) {
              if (document.getElementById(trial.iframe_id).contentWindow['final_data'].length > 0) {
                  if ($('#jspsych-survey-likert-next').is(':disabled')) {
                      $('#jspsych-survey-likert-next').removeAttr('disabled');
                      primaryEndTime = performance.now();
                      console.log("primaryEndTime recorded");
                      var primary_rt = primaryEndTime - primaryStartTime;
                      primary_response.primary_rt = primary_rt;
                  }
              }
          } else {
              if (trial.question_type == 'training_video' && watched_video == false) {
                  return
              } else {
                  $('#jspsych-survey-likert-next').removeAttr('disabled');
              }
          }
      }

      // function to handle responses by the subject
      function after_primary_response(choice) {

        // measure rt
        set_rt_and_enable_submit();

        // after a valid response, the stimulus will have the CSS class 'responded'
        // which can be used to provide visual feedback that a response was recorded
        //display_element.querySelector('#jspsych-html-button-response-stimulus').className += ' responded';
        primary_response.button = choice;

        //disable all the buttons after a response
        var btns = document.querySelectorAll('.jspsych-html-button-response-button button');

        for(var i=0; i<btns.length; i++){
          if (i != choice) {
            btns[i].removeAttribute('disabled')
          }
        }
        //btns[choice].setAttribute('disabled', 'true');
        btns[choice].setAttribute('disabled', 'true');
      };


      if (trial.textinput != null) {
        var txtbx = document.getElementById("textarea")
        $(txtbx).on("keydown", function (e) {
          set_rt_and_enable_submit()
        })
      }

      if (trial.iframe_id != null) {
        var iframe = document.getElementById(trial.iframe_id)
        $(iframe).load(function(){

            $(this).contents().find("body").on('mouseup', function(e) {
                set_rt_and_enable_submit()
            });

            $(this).contents().find("body").on('mousedown', function(e) {
                set_rt_and_enable_submit()
            });

            iframe.contentWindow.document.addEventListener("game-completed", function(e) {
                set_rt_and_enable_submit()
            });

            // no need for participant to do anything in order to continue
            if (trial.domain == 'sandbox' && trial.continue_condition == 'free_play') {
                $('#jspsych-survey-likert-next').removeAttr('disabled');
            }
        });
        var curr_ifrm = false
      }

    // PREVENT SEEKING IN VIDEOS
    // https://jsfiddle.net/o13zcrxf/4/
      var watched_video = false
      stimulus_video_loops = 0
      if (trial.question_type == 'training_video'){
          v = document.getElementById('stimulus')

          // one video in the case of LABELS
          v.onloadedmetadata = function() {
              v_curr_time = 0
              v.onseeking = function(){
                  var delta = v.currentTime - v_curr_time;
                  if (Math.abs(delta) > 0.01) {
                      console.log("Seeking is disabled");
                      v.currentTime = v_curr_time;
                  }
              }
              v.ontimeupdate = function() {
                  if (!v.seeking) {
                      v_curr_time = v.currentTime;
                  }
              }

              v.onended = function() {
                  v_curr_time = 0
                  stimulus_video_loops += 1
                  watched_video = true
                  set_rt_and_enable_submit()
              }
          }
      }

        // kill any remaining setTimeout handlers
        //jsPsych.pluginAPI.clearAllTimeouts();
    };

  return plugin;
})();

