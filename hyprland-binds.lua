-- Push-to-talk dictation binds for Hyprland's Lua config
-- Paste these into the KEYBINDINGS section of ~/.config/hypr/hyprland.lua, then `hyprctl reload`

local home = assert(os.getenv("HOME"))
local dictation = home .. "/.local/bin/dictation-client"

-- A reload registers this hook; it runs on the next Hyprland session startup
hl.on("hyprland.start", function()
    hl.exec_cmd(home .. "/Documents/GitHub/Dictation-tool/bin/dictation-session-start")
end)

hl.bind("SUPER + D", hl.dsp.exec_cmd(dictation .. " start"),
        { description = "Dictation: push to talk" })

-- Keep the release active when Super or another key is released before D
hl.bind("SUPER + D", hl.dsp.exec_cmd(dictation .. " stop"),
        { release = true, transparent = true, description = "Dictation: release to transcribe" })
