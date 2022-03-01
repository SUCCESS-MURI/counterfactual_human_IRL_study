/* jspsych-instructions.js
 * Josh de Leeuw
 *
 * This plugin displays text (including HTML formatted strings) during the experiment.
 * Use it to show instructions, provide performance feedback, etc...
 *
 * Page numbers can be displayed to help with navigation by setting show_page_number
 * to true.
 *
 * documentation: docs.jspsych.org
 *
 *
 */

jsPsych.plugins['sim-instructions'] = (function() {

  var plugin = {};

  jsPsych.pluginAPI.registerPreload('video-keyboard-response', 'stimulus', 'video');

  plugin.info = {
    name: 'sim-instructions',
    description: '',
    parameters: {
      pages: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Pages',
        default: undefined,
        array: true,
        description: 'Each element of the array is the content for a single page.'
      },
      key_forward: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        pretty_name: 'Key forward',
        default: 'rightarrow',
        description: 'The key the subject can press in order to advance to the next page.'
      },
      key_backward: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        pretty_name: 'Key backward',
        default: 'leftarrow',
        description: 'The key that the subject can press to return to the previous page.'
      },
      allow_backward: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Allow backward',
        default: true,
        description: 'If true, the subject can return to the previous page of the instructions.'
      },
      allow_keys: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Allow keys',
        default: true,
        description: 'If true, the subject can use keyboard keys to navigate the pages.'
      },
      show_clickable_nav: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Show clickable nav',
        default: false,
        description: 'If true, then a "Previous" and "Next" button will be displayed beneath the instructions.'
      },
      show_page_number: {
          type: jsPsych.plugins.parameterType.BOOL,
          pretty_name: 'Show page number',
          default: true,
          description: 'If true, and clickable navigation is enabled, then Page x/y will be shown between the nav buttons.'
      },
      button_label_previous: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label previous',
        default: 'Previous',
        description: 'The text that appears on the button to go backwards.'
      },
      button_label_next: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label next',
        default: 'Next',
        description: 'The text that appears on the button to go forwards.'
      }
    }
  }

  plugin.trial = function(display_element, trial) {

    var current_page = 0;

    var view_history = [];

    var watched_vid_arr = new Array(trial.pages.length).fill(false);
    var video_lengths_arr = new Array(trial.pages.length).fill(0.0);

    var video_loops_arr = new Array();
    var video_curr_times_arr = new Array();

    for(var i=0; i < trial.pages.length; i++) {
      video_loops_arr.push([]);
      video_curr_times_arr.push([]);
    }

    var start_time = performance.now();

    var last_page_update_time = start_time;

    function btnListener(evt){
    	evt.target.removeEventListener('click', btnListener);
    	if(this.id === "jspsych-instructions-back"){
    		back();
    	}
    	else if(this.id === 'jspsych-instructions-next'){
    		next();
    	}
    }

    function show_current_page() {
      var html = trial.pages[current_page];

      var pagenum_display = "";
      if(trial.show_page_number) {
          pagenum_display = "<span style='margin: 0 1em;' class='"+
          "jspsych-instructions-pagenum'>Page "+(current_page+1)+"/"+trial.pages.length+"</span>";
      }

      if (trial.show_clickable_nav) {

        var nav_html = "<div class='jspsych-instructions-nav' style='padding: 10px 0px;'>";
        if (trial.allow_backward) {
          var allowed = (current_page > 0 )? '' : "disabled='disabled'";
          nav_html += "<button id='jspsych-instructions-back' class='jspsych-btn' style='margin-right: 5px;' "+allowed+">&lt; "+trial.button_label_previous+"</button>";
        }
        if (trial.pages.length > 1 && trial.show_page_number) {
            nav_html += pagenum_display;
        }
        if (watched_vid_arr[current_page] == false) {
          nav_html += "<button id='jspsych-instructions-next' class='jspsych-btn'"+
              "style='margin-left: 5px;' "+"disabled='disabled'"+ ">"+trial.button_label_next+
              " &gt;</button></div>";
        } else {
          nav_html += "<button id='jspsych-instructions-next' class='jspsych-btn'"+
              "style='margin-left: 5px;'>"+trial.button_label_next+
              " &gt;</button></div>";
        }

        html += nav_html;
        display_element.innerHTML = html;
        if (current_page != 0 && trial.allow_backward) {
          display_element.querySelector('#jspsych-instructions-back').addEventListener('click', btnListener);
        }

        display_element.querySelector('#jspsych-instructions-next').addEventListener('click', btnListener);
      } else {
        if (trial.show_page_number && trial.pages.length > 1) {
          // page numbers for non-mouse navigation
          html += "<div class='jspsych-instructions-pagenum'>"+pagenum_display+"</div>"
        }
        display_element.innerHTML = html;
      }

      v = document.getElementById('stimulus')

      v.onloadedmetadata = function() {
        console.log("onloadedmetadata")
        v_curr_time = 0

        current_video_loops = 0

        if (watched_vid_arr[current_page] == false) {
          v.onseeking = function () {
            console.log("seeking")
            var delta = v.currentTime - v_curr_time;
            if (Math.abs(delta) > 0.01) {
              console.log("Seeking is disabled");
              v.currentTime = v_curr_time;
            }
          }
          v.ontimeupdate = function () {
            if (!v.seeking) {
              v_curr_time = v.currentTime;
            }
          }
        }
          v.onended = function() {
            v_curr_time = 0
            current_video_loops += 1
            console.log("incremented current video loops to:")
            console.log(current_video_loops)
            // after at least one full watch through, enable the 'next' button and also enable seeking
            watched_vid_arr[current_page] = true
            if ($('#jspsych-instructions-next').is(':disabled')) {
              $('#jspsych-instructions-next').removeAttr('disabled');
              v.onseeking = function (){}
              v.ontimeupdate = function (){}
            }
            console.log("onended")
          }

      }
    }

    function next() {

      add_current_page_to_view_history()

      // if the video length hasn't been initialized yet, initialize it
      if (video_lengths_arr[current_page] == 0.0) {
        video_lengths_arr[current_page] = v.duration
      }

      v = document.getElementById('stimulus')
      video_loops_arr[current_page].push(current_video_loops);
      video_curr_times_arr[current_page].push(v.currentTime);

      current_page++;

      // if done, finish up...
      if (current_page >= trial.pages.length) {
        endTrial();
      } else {
        show_current_page();
      }

    }

    function back() {

      add_current_page_to_view_history()

      // if the video length hasn't been initialized yet, initialize it
      if (video_lengths_arr[current_page] == 0.0) {
        video_lengths_arr[current_page] = v.duration
      }

      v = document.getElementById('stimulus')
      video_loops_arr[current_page].push(current_video_loops);
      video_curr_times_arr[current_page].push(v.currentTime);

      current_page--;

      show_current_page();
    }

    function add_current_page_to_view_history() {

      var current_time = performance.now();

      var page_view_time = current_time - last_page_update_time;

      view_history.push({
        page_index: current_page,
        viewing_time: page_view_time
      });

      last_page_update_time = current_time;
    }

    function endTrial() {

      if (trial.allow_keys) {
        jsPsych.pluginAPI.cancelKeyboardResponse(keyboard_listener);
      }

      display_element.innerHTML = '';

      var trial_data = {
        // Primary information
        "primary_stimulus": JSON.stringify(view_history),
        "primary_rt": performance.now() - start_time,
        "primary_answer": {"video_lengths": video_lengths_arr,
          "video_loops": video_loops_arr,
          "video_curr_times": video_curr_times_arr},

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

      jsPsych.finishTrial(trial_data);
    }

    var after_response = function(info) {

      // have to reinitialize this instead of letting it persist to prevent accidental skips of pages by holding down keys too long
      keyboard_listener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: [trial.key_forward, trial.key_backward],
        rt_method: 'performance',
        persist: false,
        allow_held_key: false
      });
      // check if key is forwards or backwards and update page
      if (jsPsych.pluginAPI.compareKeys(info.key, trial.key_backward)) {
        if (current_page !== 0 && trial.allow_backward) {
          back();
        }
      }

      if (jsPsych.pluginAPI.compareKeys(info.key, trial.key_forward)) {
        next();
      }

    };

    show_current_page();

    if (trial.allow_keys) {
      var keyboard_listener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: [trial.key_forward, trial.key_backward],
        rt_method: 'performance',
        persist: false
      });
    }
  };

  return plugin;
})();
