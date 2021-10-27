" Config
set tabstop=2
set shiftwidth=2

" vim-cmake configuration
let s:bin_folder = "bin.debug"
let s:runner = "test"
let s:args = ""
" let s:before_script = "activate_run.sh"
let s:before_script = ""
let s:conan_args = ""

let g:cmake_config = "Debug"
let g:cmake_build_dir_location = "build"
let g:cmake_link_compile_commands = 1
let g:cmake_generate_options = ["-DBUILD_TESTS=ON"]
let g:cmake_build_options = ["--target", s:runner]

" Key bindings
map <leader>cx :CMakeClean<CR>
map <leader>cs :execute 'CMakeSwitch' g:cmake_config<CR>
map <leader>cc :execute 'CMakeGenerate' g:cmake_config<CR>
map <leader>cb :CMakeBuild<CR>
map <leader>cr :Run<CR>
map <leader>c; :BuildRun<CR>
map <leader>cp :ConanInstall<CR>

" Get build directory name
function! BuildDir()
  return g:cmake_build_dir_location . '/' . g:cmake_config
endfunction

" Conan install
function! ConanInstall()
  let l:command = ['conan', 'install', '-if', BuildDir(), '.', s:conan_args]

  call cmake#console#SetCmdId('conan')
  call cmake#command#Run(l:command, 0, 0)
endfunction

command! ConanInstall call ConanInstall()

" Source before script
function! SourceScript()
  return ['source', BuildDir() . '/' . s:before_script]
endfunction

" Go to bin directory
function! GoToBinDir()
  return ['cd', s:bin_folder]
endfunction

" Launch runner
function! LaunchRunner()
  return ['./' . s:runner, s:args]
endfunction

" Execute Run command in a vim-cmake console
function! ExecuteRun()
  let l:command = ['(']

  if s:before_script != ""
    let l:command += SourceScript()
    let l:command += ['&&']
  endif

  let l:command += GoToBinDir()
  let l:command += ['&&']
  let l:command += LaunchRunner()
  let l:command += [')']

  call cmake#console#SetCmdId('run')
  call cmake#command#Run(l:command, 0, 0)
endfunction

" Run
function! Run()
    call ExecuteRun()
endfunction

command! Run call Run()

" Build & Run
function! BuildRun()
    execute 'CMakeBuild'
    call Run()
endfunction

command! BuildRun call BuildRun()

