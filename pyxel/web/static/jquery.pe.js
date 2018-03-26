$.fn.extend({
    highlight: function(text, ignore_if_equal) {
        var ind = $('.indicator', $(this));
        if (ignore_if_equal && ind.text() == text) {
            return $(this);
        }
        ind.css('background-color', '')
        ind.text(text).prop('title', text).addClass('indicator-hilite');
        setTimeout(function() { ind.removeClass('indicator-hilite'); }, 500);
        return $(this);
    },

    progress: function(enabled, color) {
        var ind = $('.indicator', $(this));
        var run = $('.pe-progress .pe-progress-running');
        run.removeClass('pe-progress-running').css('background-color', '')
        if (color) {
            ind.css('background-color', color);
        }
        if (enabled) {
            ind.addClass('pe-progress-running');
        }
    },

    set_enabled: function(enabled) {
        var disabled = enabled ? false : 'disabled';
        $('select', $(this)).prop('disabled', disabled);
        $('input[type="text"]', $(this)).prop('disabled', disabled);
    },

    is_enabled: function() {
        return $('input[type="checkbox"]', $(this)).is(":checked");
    },

    set_button_text: function(text) {
        var btn = $('button', this);
        btn.text(text);
        return $(this);
    },

    get_context: function() {
        var context = $(this);
        if (context.prop('tagName') != 'TR') {
            context = $(this).parents('tr').first();
        }
        if (!context) {
            console.log('ERROR: incorrect context at element:', $(this).html());
        }
        return context;
    },

    get_value: function() {
        var context = $(this).get_context();
        var value = null;
        if ($('select', context).length) {
            value = $('select', context).val();
        } else if ($('input', context).length == 1) {
            value = $('input', context).val();
        } else if ($('input', context).length > 1) {
            value = [];
            $('input', context).each(function() {
                value.push($(this).val());
            });
        }
        return value;
    },

    set_value: function(value, use_tr_context) {
        var context = $(this);
        if (use_tr_context) {
            context = $(this).get_context();
        }

        if ($('select', context).length) {
            $('select', context).val(value);

//        } else if ($('input[type=checkbox]', context).length == 1) {
//            $('input[type=checkbox]', context).prop('checked', value);

        } else if ($('input', context).length == 1) {
            $('input', context).val(value);

        } else if ($('input', context).length > 1) {
            $('input', context).each(function(index) {
                $(this).val(value[index]);
            });
        }
        return $(this);
    }
});