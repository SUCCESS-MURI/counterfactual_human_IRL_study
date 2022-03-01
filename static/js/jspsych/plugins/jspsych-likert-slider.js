/**
 * jspsych-html-button-response
 * Josh de Leeuw
 *
 * plugin for displaying a stimulus and getting a keyboard response
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["likert-slider"] = (function() {

    var plugin = {};
  
    jsPsych.pluginAPI.registerPreload('video-keyboard-response', 'stimulus', 'video');
  
  
    plugin.info = {
      name: 'likert-slider',
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
        preamble: {
            type: jsPsych.plugins.parameterType.STRING,
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
  


        //SLIDER RESPONSE 
        min: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Min slider',
            default: 0,
            description: 'Sets the minimum value of the slider.'
        },
        max: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Max slider',
            default: 100,
            description: 'Sets the maximum value of the slider',
        },
        start: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Slider starting value',
            default: 50,
            description: 'Sets the starting value of the slider',
        },
        step: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name: 'Step',
            default: 1,
            description: 'Sets the step of the slider'
        },
        labels: {
            type: jsPsych.plugins.parameterType.HTML_STRING,
            pretty_name:'Labels',
            default: [],
            array: true,
            description: 'Labels of the slider.',
        },
        slider_width: {
            type: jsPsych.plugins.parameterType.INT,
            pretty_name:'Slider width',
            default: null,
            description: 'Width of the slider in pixels.'
        },
        require_movement: {
            type: jsPsych.plugins.parameterType.BOOL,
            pretty_name: 'Require movement',
            default: false,
            description: 'If true, the participant will have to move the slider before continuing.'
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
        }
      }         
    }
  
    plugin.trial = function(display_element, trial) {
  
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

        
        //    html = '<div id="jspsych-html-slider-response-wrapper" style="margin: 100px 0px;">';
            html += '<div id="jspsych-html-slider-response-stimulus">' + trial.stimulus + '</div>';
          html += '<div class="jspsych-html-slider-response-container" style="position:relative; margin: 0 auto 3em auto; ';
          if(trial.slider_width !== null){
            html += 'width:'+trial.slider_width+'px;';
          }
          html += '">';

          html += '<input type="range" value="'+trial.start+'" min="'+trial.min+'" max="'+trial.max+'" step="'+trial.step+'" style="width: 100%;" id="jspsych-html-slider-response-response"></input>';
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

      
  