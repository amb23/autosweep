#!/bin/bash

## FIXME: should we use git config --global when setting values

function usage() {
    cat << END
Usage: setup.sh [-g] [-u] [-a] [-r] [-x]

Options:
    -g      Apply all changes to user-global config instead of per-repository config

    -u      Setup user name/email
    -a      Setup aliases
    -r      Set misc config flags
    -x      Setup url rewriting rules for pushing into repo-ln


Does a basic setup of the git checkout.
It will verify & set your name/email, add a bunch of convenience aliases
and tune a few things in git repo.

Will apply all setting unless a subset of settings is requested explicitly
END
}

function git_version_check() {
    local v=($(git --version))

    if [[ "${#v[@]}" -ne 3 ]]; then
        echo "=== WARN: unexpected git --version output: ${v[@]}" >&2
        return
    fi
    log "System git version: ${v[@]}"

    # stupid hack: user sort -n -t. to order version strings
    local older=$(echo -e "1.7.11\n${v[2]}" | sort -n | head -n1)
    log "Sorted (older) version: ${older}"

    if [[ "$older" != "1.7.11" ]]; then
        # system git is older than 1.7.11
        # print warning, and don't use push.default = simple
        echo "==== WARN: it seems that your git version is quite old: ${v[2]}" >&2
        echo "==== WARN: consider upgrading to at least 1.8.xx"
        push_simple_supported=n
    else
        push_simple_supported=y
    fi
}

function set_user_id() {
    local name email new_name new_email

    name=$(git config user.name)
    email=$(git config user.email)
    log "Existing name/email: $name <$email>"

    if [[ -n "$name" && -n "$email" ]]; then
        echo "Your username is already set to \"${name} <${email}>\""
        while read -p 'Want to change it [y/N]: '; do
            case "$REPLY" in
                ""|[nN]*)
                    return 0 ;;
                [yY]*)
                    break ;;
            esac
        done
    fi

    while true; do
        read -p "Enter your name [$name]: " new_name
        read -p "Enter your email [$email]: " new_email
        if [[ -z "$new_name" ]]; then
            new_name="$name"
        fi
        if [[ -z "$new_email" ]]; then
            new_email="$email"
        fi

        log "New name/email: $new_name <$new_email>"

        if [[ -n "$new_name" && -n "$new_email" ]]; then
            log "Setting name/email: $new_name <$new_email>"
            gset user.name "$new_name" && gset user.email "$new_email"
            return
        fi
    done
}

function repo_setup() {
    gset color.ui         auto
    ## longer abbreviated SHAs
    gset core.abbrev      12
    ## detect renames & copied files when showing diffs
    gset diff.renames     copy
    ## use more mnemonic prefixes in diffs:
    ## (i)ndex, (w)orking tree, (c)ommit, (o)bject
    gset diff.mnemonicprefix  true
    gset gc.pruneexpire   6.month.ago
    gset gc.reflogexpire  6.month.ago
    gset gc.reflogexpireunreachable   6.month.ago
    ## less aggressive copy detection in blame
    ## default settings are way too slow for our repo
    gset gui.fastcopyblame    true
    ## use date format suitable for rev timestamp in jira
    gset log.date         iso
    ## much more aggressive rename detection in merges
    ## might happen when you merge stuff between releases&trunk
    gset merge.renameLimit 10000
    gset pack.compression 9
    ## disable pager for short logs
    ## this prevents it from firing on long commit messages
    gset pager.ll         false
    gset pager.la         false
    gset pager.lanc       false
    gset pager.jira       false
    ## default push policy
    if [[ "$push_simple_supported" = "y" ]]; then
        gset push.default     simple
    else
        gset push.default     upstream
    fi
    ## validate checksums when transferring data
    gset transfer.fsckObjects true
    # default to git pull --ff-only
    # to discourange ppl from accidentally creating merge commits when they probably want to rebase instead
    gset pull.ff only
}

function set_urls() {
    gset 'url.repo-ln.pushInsteadOf' 'repo-au'
    gset --add 'url.repo-ln.pushInsteadOf' 'repo-us'
    gset 'url.ssh://repo-ln.pushInsteadOf' 'ssh://repo-au'
    gset --add 'url.ssh://repo-ln.pushInsteadOf' 'ssh://repo-us'
}

function set_aliases() {
    ## custom log formats
    gset pretty.la    'tformat:%C(yellow)%h%Creset%C(bold red)%d%Creset %s %C(bold blue)<%an>%Creset %C(green)%ad%Creset'
    gset pretty.la-nc 'tformat:%h%d %s <%an> %ad'

    ## few different 1-line-per-commit log formats
    gset alias.ll 'log --oneline --decorate -20'
    gset alias.la 'log --pretty=la --decorate --date=short -20'
    gset alias.lanc 'log --pretty=la-nc --decorate --date=short -20'
	## visual tree for git
	gset alias.lt 'log --oneline --decorate --all --graph --simplify-by-decoration'

    ## log revisions constituting a commit (revisions brought in by a merge)
    ## essentially git log rev^..rev
    ## git (logc|lc) <revision_id>
    gset alias.logc '!f() { local r a f; r=`git rev-parse --revs-only "$@"` && r=${r:-HEAD} && a=`git rev-parse --sq --no-revs --flags "$@"` && f=`git rev-parse --sq --no-revs --no-flags "$@"` && cd ./$GIT_PREFIX && eval "git log $a $r^..$r -- $f"; }; f'
    gset alias.lc '!f() { local r a f; r=`git rev-parse --revs-only "$@"` && r=${r:-HEAD} && a=`git rev-parse --sq --no-revs --flags "$@"` && f=`git rev-parse --sq --no-revs --no-flags "$@"` && cd ./$GIT_PREFIX && eval "git ll $a $r^..$r -- $f"; }; f'
    ## diff changes introduced by commit (brought in by a merge)
    ## essentially git diff rev^..rev
    ## git diffc <revision_id>
    gset alias.diffc '!f() { local r a f; r=`git rev-parse --revs-only "$@"` && r=${r:-HEAD} && a=`git rev-parse --sq --no-revs --flags "$@"` && f=`git rev-parse --sq --no-revs --no-flags "$@"` && cd ./$GIT_PREFIX && eval "git diff $a $r^..$r -- $f"; }; f'
    ## log incoming/outgoing revision relative to specified ref (or upstream/tracking)
    ## essentially git log ..@{u}   # for incoming
    ## or          git log @{u}..   # for outgoing
    ## git (loginc|linc|logout|lout) [<ref>]
    gset alias.loginc '!f() { local r a f; r=`git rev-parse --revs-only "$@"` && r=${r:-'\''@{u}'\''} && a=`git rev-parse --sq --no-revs --flags "$@"` && f=`git rev-parse --sq --no-revs --no-flags "$@"` && cd ./$GIT_PREFIX && eval "git log $a ..$r -- $f"; }; f'
    gset alias.linc '!f() { local r a f; r=`git rev-parse --revs-only "$@"` && r=${r:-'\''@{u}'\''} && a=`git rev-parse --sq --no-revs --flags "$@"` && f=`git rev-parse --sq --no-revs --no-flags "$@"` && cd ./$GIT_PREFIX && eval "git ll $a ..$r -- $f"; }; f'
    gset alias.logout '!f() { local r a f; r=`git rev-parse --revs-only "$@"` && r=${r:-'\''@{u}'\''} && a=`git rev-parse --sq --no-revs --flags "$@"` && f=`git rev-parse --sq --no-revs --no-flags "$@"` && cd ./$GIT_PREFIX && eval "git log $a $r.. -- $f"; }; f'
    gset alias.lout '!f() { local r a f; r=`git rev-parse --revs-only "$@"` && r=${r:-'\''@{u}'\''} && a=`git rev-parse --sq --no-revs --flags "$@"` && f=`git rev-parse --sq --no-revs --no-flags "$@"` && cd ./$GIT_PREFIX && eval "git ll $a $r.. -- $f"; }; f'
    ## diff incoming/outgoing changes relative to specified ref (or upstream/tracking)
    ## essentially git diff ...@{u}   # for incoming
    ## or          git diff @{u}...   # for outgoing
    ## git (diffinc|diffout) [<ref>]
    gset alias.diffinc '!f() { local r a f; r=`git rev-parse --revs-only "$@"` && r=${r:-'\''@{u}'\''} && a=`git rev-parse --sq --no-revs --flags "$@"` && f=`git rev-parse --sq --no-revs --no-flags "$@"` && cd ./$GIT_PREFIX && eval "git diff $a ...$r -- $f"; }; f'
    gset alias.diffout '!f() { local r a f; r=`git rev-parse --revs-only "$@"` && r=${r:-'\''@{u}'\''} && a=`git rev-parse --sq --no-revs --flags "$@"` && f=`git rev-parse --sq --no-revs --no-flags "$@"` && cd ./$GIT_PREFIX && eval "git diff $a $r... -- $f"; }; f'
    ## unstage a file (or all of them)
    gset alias.unstage 'reset HEAD --'
    ## abandon current changes and check-out version from the index
    ## if there is unresolved merge -- check it out with diff3 style (with base)
    ## git remerge <file>
    gset alias.remerge 'checkout -m --conflict=diff3 --'
    ## short status
    gset alias.st 'status -sb'
    ## shorter aliases for commit & checkout
    gset alias.ci 'commit'
    gset alias.co 'checkout'
    ## diff of the staged changes
    gset alias.ix 'diff --cached'
    ## add currently staged changes to last commit
    gset alias.fixup 'commit --amend -C HEAD'
    ## update all remotes
    gset alias.rup 'remote -v update'
    ## find file by name
    ## git f quoter
    gset alias.f '!git ls-files -c -o --exclude-standard | grep --color=auto -i'
    ## show aliases
    gset alias.aliases '!git config -l | grep ^alias\. | cut -c 7-'
    ## show upstream branches
    gset alias.info 'for-each-ref --format="%(refname:short)  <=  %(upstream:short)" refs/heads'
    ## commit info for jira
    gset alias.jira '!f() { local r=${1:-HEAD}; git log -1 --pretty="tformat:Id:      $(git describe --abbrev=12 --long $r)%nDate:    %ci%nAuthor:  %an <%ae>%w(70,4,4)%n%n%-s%n%n%-b" $r; }; f'
}

function log() {
    if [[ "x$DEBUG" != "x" ]]; then
        echo "=== $@"
    fi
}

function gset() {
echo "Setting $1"
git config $gitconfig_flags "$@"
}


#user_id
#set_aliases
#repo_setup

########################
# main
gitconfig_flags=''

while getopts "huarxg" opt; do
    case "$opt" in
        '?'|h)
            usage
            exit 1 ;;
        u)
            do_set_user_id=1 ;;
        a)
            do_set_aliases=1 ;;
        r)
            do_repo_setup=1 ;;
        x)
            do_url_setup=1 ;;
        g)
            gitconfig_flags="--global"
    esac
done


do_all=1
if [[ -n "$do_set_user_id" || -n "$do_set_aliases" || -n "$do_repo_setup" || -n "$do_url_setup" ]]; then
    unset do_all
fi

git_version_check

if [[ -n "$do_set_user_id" || -n "$do_all" ]]; then
    set_user_id
fi
if [[ -n "$do_set_aliases" || -n "$do_all" ]]; then
    set_aliases
fi
if [[ -n "$do_repo_setup" || -n "$do_all" ]]; then
    repo_setup
fi
if [[ -n "$do_url_setup" || -n "$do_all" ]]; then
    set_urls
fi
