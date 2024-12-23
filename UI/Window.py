import wx

def show_window():
    app = wx.App()
    frm = wx.Frame(None, title = "AI App")
    frm.Show()
    app.MainLoop()