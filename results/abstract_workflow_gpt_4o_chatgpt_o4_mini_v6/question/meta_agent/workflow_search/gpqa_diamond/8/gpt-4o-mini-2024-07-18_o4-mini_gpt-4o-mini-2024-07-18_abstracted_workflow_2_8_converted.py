async def forward_8(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    cot_instruction = "Sub-task 1: Analyze the provided molecules to identify their structural characteristics and symmetry properties, focusing on the concept of C3h symmetry."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing molecules for C3h symmetry, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    cot_sc_instruction = "Sub-task 2: Classify the molecules based on their symmetry properties, determining which of them may exhibit C3h symmetry."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, classifying molecules for C3h symmetry, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_final, answer_final = await final_decision_agent([taskInfo] + possible_answers, 
                                                 "Sub-task 2: Make final decision on which molecules exhibit C3h symmetry.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, determining C3h symmetry, thinking: {thinking_final.content}; answer: {answer_final.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_final.content}; answer - {answer_final.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_final, answer_final, sub_tasks, agents)
    return final_answer