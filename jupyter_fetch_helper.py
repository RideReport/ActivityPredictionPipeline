def _getUrlWithJWT(url, token):
    import requests
    headers = {
        'Authorization': 'JWT {}'.format(token),
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def fetchResourceInJupyter(varname, url):
    from IPython.display import HTML, Javascript, display_html, display_javascript
    import base64
    import json

    # Get a token from browser-land so we can make a request in python-land and fetch stuff directly that way.
    func = """
    (function(varname, url) {
        function onDone(msg) {
            var content = msg.content;
            var status = content.status;
            if (status == 'ok') {
                element.html('<tt>'+ varname +'</tt> loaded successfully');
            }
            else if (status == 'error') {
                element.html('<tt>Error: '+ content.ename +' '+ content.evalue +'</tt>');
            }
            else {
                element.html('<tt>Unknown error: '+ content.status +'</tt>');
            }
        }

        var xhr = new XMLHttpRequest();
        xhr.open('POST', 'https://ride.report/api/v2/shortlived_jwt_token');
        xhr.withCredentials = true;
        xhr.onload = function(e) {
            if (xhr.status != 200) {
                element.html('<tt>Failed: '+ xhr.status +' '+ xhr.responseText +'</tt>');
                return;
            }
            var kernel = IPython.notebook.kernel;
            var token = JSON.parse(xhr.responseText)['token'];
            var callbacks = {
                shell: {
                    reply: onDone,
                }
            };
            kernel.execute('import jupyter_fetch_helper; '+ varname +' = jupyter_fetch_helper._getUrlWithJWT("' + url + '", "' + token + '")', callbacks);
        };
        xhr.send();
        element.html('Loading ...');
    })"""
    display_javascript(Javascript(func + '("{}", "{}");'.format(varname, url)))

def fetchTSDInJupyter(arg):
    url = 'https://ride.report/__tools/inspect/tripsensordata_raw/{}'.format(arg)
    fetchResourceInJupyter('tsd_dict', url)

def fetchASDInJupyter(arg):
    url = 'https://ride.report/__tools/inspect/androidsensordata_raw/{}'.format(arg)
    fetchResourceInJupyter('asd_dict', url)

def fetchICDInJupyter(arg):
    url = 'https://ride.report/__tools/inspect/iosclassificationdata_raw/{}'.format(arg)
    fetchResourceInJupyter('icd_dict', url)
