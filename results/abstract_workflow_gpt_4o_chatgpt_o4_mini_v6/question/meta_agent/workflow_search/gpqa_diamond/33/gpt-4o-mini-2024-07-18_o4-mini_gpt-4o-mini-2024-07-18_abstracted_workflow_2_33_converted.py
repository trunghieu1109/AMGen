async def forward_33(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    cot_instruction = "Sub-task 1: Analyze the provided chemical reactions and identify the starting materials (vicinal diols) and the acid catalyst involved in the Pinacol rearrangement."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing chemical reactions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    cot_sc_instruction = "Sub-task 2: Classify the three given compounds based on their structural features and functional groups relevant to the Pinacol rearrangement."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, classifying compounds, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinkingmapping}; answer - {possible_answers}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    cot_instruction3 = "Sub-task 3: Determine the expected products of the Pinacol rearrangement for each of the classified compounds, considering the mechanism of carbocation formation and group shifting."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, possible_answers], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, determining expected products, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    debate_instruction = "Sub-task 4: Evaluate the provided choices for the products of the rearrangement against the expected products derived in subtask 3, and select the correct options."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    all_thinking4 = [[] for _ in range(N_max)]
    all_answer4 = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, answer3], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating choices, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_final, answer_final = await final_decision_agent([taskInfo] + all_thinking4[-1] + all_answer4[-1], 
                                                 "Sub-task 5: Make final decision on the correct product choices.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision, thinking: {thinking_final.content}; answer: {answer_final.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_final.content}; answer - {answer_final.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_final, answer_final, sub_tasks, agents)
    return final_answer