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
var position = null;
var primary_misc = {}

jsPsych.plugins["likert-pref"] = (function() {

  var plugin = {};

  jsPsych.pluginAPI.registerPreload('video-keyboard-response', 'stimulus', 'video');


  plugin.info = {
    name: 'likert-pref',
    description: '',
    parameters: {

      //SECONDARY CHECK
      sources: {
        type: jsPsych.plugins.parameterType.VIDEO,
        pretty_name: 'Video',
        default: undefined,
        description: 'The video file to play.'
      },
      checkChoices: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        pretty_name: 'Choices',
        array: true,
        default: jsPsych.ALL_KEYS,
        description: 'The keys the subject is allowed to press to respond to the stimulus.'
      },
      width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Width',
        default: '',
        description: 'The width of the video in pixels.'
      },
      height: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Height',
        default: '',
        description: 'The height of the video display in pixels.'
      },
      autoplay: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Autoplay',
        default: true,
        description: 'If true, the video will begin playing as soon as it has loaded.'
      },
      controls: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Controls',
        default: false,
        description: 'If true, the subject will be able to pause the video or move the playback to any point in the video.'
      },

      //PRIMARY QUESTION STIMULUS + QUESTION
      stimulus: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: undefined,
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
      iframe_position: {
        type: jsPsych.plugins.parameterType.DOUBLE,
        pretty_name: 'Lunar Lander position',
        default: null,
        description: 'The starting x coordinate of lunar lander'
      },
      preamble: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Preamble',
        default: null,
        description: 'String to display at top of the page.'
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
        default: undefined,
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
      interaction_type: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Interaction Type',
        default: null,
        description: 'Identifies what type of primary question was asked.'
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
      }
    }
  }

  plugin.trial = function(display_element, trial) {
    position = trial.iframe_position
    if (trial.iframe_src != null) {
      path = trial.iframe_src
    }
    num_button_toggles = 0
    primary_misc = {}

  /* START SECONDARY TASK */

        // setup stimulus
      var html = '<div>'
      html += '<video id="jspsych-video-keyboard-response-stimulus"';

      if(trial.width) {
        html += ' width=160';
      }
      if(trial.height) {
        html += ' height=40';
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


      var secondaryStartTime = performance.now()

      var video_loops = 0
      var reset = function() {
          video_loops += 1
          vid.currentTime = 0
          vid.play()
      }

      /* END SECONDARY TASK */


    /* START PRIMARY QUESTION */
    // show preamble text
    if(trial.preamble !== null){
      html += '<div id="jspsych-html-button-response-preamble" class="jspsych-html-button-response-preamble">'+trial.preamble+'</div>';
    }

    // display stimulus
    html += '<div id="jspsych-html-button-response-stimulus">'+trial.stimulus+'</div>';
    if (trial.caption !== null){
      html += '<p>' + trial.caption + '</p>'
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

    //show prompt if there is one
    if (trial.prompt !== null) {
      html += trial.prompt;
    }

    //display_element.innerHTML = html;

    /* END PRIMARY QUESTION */


    /* START CONFIDENCE SCALE */

    if(trial.scale_width !== null){
      var w = trial.scale_width + 'px';
    } else {
      var w = '100%';
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

      // start time
      var primaryStartTime = performance.now();
      if(trial.showButtons){
        // add event listeners to buttons
        for (var i = 0; i < trial.choices.length; i++) {
          display_element.querySelector('#jspsych-html-button-response-button-' + i).addEventListener('click', function(e){
            if (trial.domain == '0' && watched_video == false) {
              return
            }
            var choice = e.currentTarget.getAttribute('data-choice'); // don't use dataset for jsdom compatibility
            after_primary_response(choice);
            num_button_toggles += 1
          });
        }
      }

    display_element.querySelector('#jspsych-survey-likert-form').addEventListener('submit', function(e){
      e.preventDefault();
      // measure response time
      var likertEndTime = performance.now();
      var response_time = likertEndTime;
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

      var vid = document.getElementById("jspsych-video-keyboard-response-stimulus");
      currVidTime = vid.currentTime;

      if (trial.domain == 0) {
        if (trial.interaction_type == 1) {
          vA = document.getElementById('stimulus_A')
          vB = document.getElementById('stimulus_B')
          primary_misc['stimulus_video_loops'] = [stimulus_A_video_loops, stimulus_B_video_loops]
          primary_misc['stimulus_vid_curr_time'] = [vA.currentTime, vB.currentTime]
        }
        if (trial.interaction_type == 2) {
          v = document.getElementById('stimulus')
          primary_misc['stimulus_video_loops'] = stimulus_video_loops
          primary_misc['stimulus_vid_curr_time'] = v.currentTime
        }
      }

      primary_misc['num_button_toggles'] = num_button_toggles

      // save data
      trial_data = {
        // Primary Information
        "primary_stimulus": trial.stimulus,
        "primary_rt": primary_response.primary_rt,
        "primary_answer": primary_response.button,
        "proposed_label": trial.caption,
        "primary_misc": primary_misc,

        // Secondary Information
        "secondary_stimulus": trial.sources,
        "secondary_key_press": secondary_response.key_press,
        "secondary_rt": JSON.stringify(secondary_response.secondary_rt),
        "vid_curr_time": vid.currentTime,
        "video_loops": video_loops,

        // Confidence Scale
        "likert_rt": response_time,
        "likert_response": JSON.stringify(question_data),
        "likert_question_order": JSON.stringify(question_order),

        // Misc.
        "question_type": trial.question_type,
        "question_id": trial.question_id,
        "interaction_type": trial.interaction_type,
        "domain": trial.domain,
        "attn_checks": null,
        "conf_code": null,
        "condition": trial.condition
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
        var primaryEndTime = performance.now();
        var primary_rt = primaryEndTime - primaryStartTime;
        primary_response.primary_rt = primary_rt;

        if(trial.iframe_id !== null) {
          if (document.getElementById(trial.iframe_id).contentWindow['final_data'].length > 0) {
            $('#jspsych-survey-likert-next').removeAttr('disabled');
          }
        } else {
          if (trial.domain == '0' && watched_video == false) {
            return
          }
          $('#jspsych-survey-likert-next').removeAttr('disabled');
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

      // TODO(pkoppol): Where are we setting secondaryStartTime and why?
      var after_secondary_response = function(info) {
        // after a valid response, the stimulus will have the CSS class 'responded'
        // which can be used to provide visual feedback that a response was recorded
        display_element.querySelector('#jspsych-video-keyboard-response-stimulus').className += ' responded';
        var vid = document.getElementById("jspsych-video-keyboard-response-stimulus");
        timestamp = vid.currentTime;
        if (secondary_response.key_press == null){
          secondary_response.key_press = 1;
          secondary_response.secondary_rt = [timestamp]
        } else {
          secondary_response.key_press +=1;
          secondary_response.secondary_rt.push(timestamp)
        }
      };

        // start the response listener
      if (trial.checkChoices != jsPsych.NO_KEYS) {
        $(document).on("keydown", function (e) {
          if (trial.checkChoices.includes(e.key)) {
            after_secondary_response();
          }
        });
      }

      if (trial.textinput != null) {
        var txtbx = document.getElementById("textarea")
        $(txtbx).on("keydown", function (e) {
          set_rt_and_enable_submit()
        })
      }

      if (trial.iframe_id != null) {
        var iframe = document.getElementById(trial.iframe_id)
        $(iframe).load(function(){
            $(this).contents().find("body").on('keydown', function(e) {
               if (trial.checkChoices.includes(e.key)) {
                  after_secondary_response();
                } else {
                  set_rt_and_enable_submit()
                }
            });

            $(this).contents().find("body").on('mouseup', function(e) {
                set_rt_and_enable_submit()
            });

            $(this).contents().find("body").on('mousedown', function(e) {
                set_rt_and_enable_submit()
            });

            iframe.contentWindow.document.addEventListener("game-completed", function(e) {
                set_rt_and_enable_submit()
            });
        });

        var curr_ifrm = false
      }

      var vid = document.getElementById("jspsych-video-keyboard-response-stimulus");
      vid.onloadedmetadata = function() {
        vid.addEventListener('ended', reset);
      }

    var watched_video = true

    // PREVENT SEEKING IN VIDEOS
    // https://jsfiddle.net/o13zcrxf/4/
    if (trial.domain == '0') {
          var watched_video = false
          stimulus_video_loops = 0
          stimulus_A_video_loops = 0
          stimulus_B_video_loops = 0
          if (trial.interaction_type == '2'){
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
                  watched_video = true
              }

              v.onended = function() {
                v_curr_time = 0
                stimulus_video_loops += 1
              }
            }
          } else if (trial.interaction_type == '1') {
            // two videos in the case of PREFERENCES
            vA = document.getElementById('stimulus_A')
            vB = document.getElementById('stimulus_B')
            vA.onloadedmetadata = function() {
              vA_curr_time = 0
              vA.onseeking = function(){
                var delta = vA.currentTime - vA_curr_time;
                if (Math.abs(delta) > 0.01) {
                  console.log("Seeking is disabled");
                  vA.currentTime = vA_curr_time;
                }
              }

              vA.ontimeupdate = function() {
                  if (!vA.seeking) {
                    vA_curr_time = vA.currentTime;
                  }
                  watched_video = true

              }

              vA.onended = function() {
                vA_curr_time = 0
                stimulus_A_video_loops += 1
              }
            }

            vB.onloadedmetadata = function() {
              vB_curr_time = 0
              vB.onseeking = function(){
                var delta = vB.currentTime - vB_curr_time;
                if (Math.abs(delta) > 0.01) {
                  console.log("Seeking is disabled");
                  vB.currentTime = vB_curr_time;
                }
              }

              vB.ontimeupdate = function() {
                  if (!vB.seeking) {
                    vB_curr_time = vB.currentTime;
                  }
                  watched_video = true
              }

              vB.onended = function() {
                vB_curr_time = 0
                stimulus_B_video_loops += 1
              }
            }
          }
        }

        // kill any remaining setTimeout handlers
        //jsPsych.pluginAPI.clearAllTimeouts();
    };

  return plugin;
})();

