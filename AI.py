# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:40:20 2019

@author: Cullen
"""

class AI():
    def get_move(self, packet):
        raise Exception("Should be implemented by child. This is the abstract")
        
        
        
#Testing
if __name__ == "__main__":
    ai = AI()
    ai.get_move([])