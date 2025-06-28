async def forward_76(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    cot_instruction = "Sub-task 1: Analyze the Cope rearrangement reaction to understand the mechanism and the expected products based on the provided reactants."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing Cope rearrangement, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    cot_instruction_2 = "Sub-task 2: Identify the specific reactants involved in the two reactions (A and B) and classify their structures and functional groups."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, identifying reactants, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    cot_sc_instruction_A = "Sub-task 3: Evaluate the possible products for reaction A by applying the Cope rearrangement mechanism to the identified reactant structure."
    N = self.max_sc
    cot_agents_A = [LLMAgentBase(["thinking", "answer"], "CoT-SC Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_A = []
    thinkingmapping_A = {}
    answermapping_A = {}
    
    for i in range(N):
        thinking3, answer3 = await cot_agents_A[i]([taskInfo, thinking2, answer2], cot_sc_instruction_A, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_A[i].id}, evaluating products for reaction A, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_A.append(answer3.content)
        thinkingmapping_A[answer3.content] = thinking3
        answermapping_A[answer3.content] = answer3
    
    sub_tasks.append(f"Sub-task 3 output: possible answers for A: {possible_answers_A}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    cot_sc_instruction_B = "Sub-task 4: Evaluate the possible products for reaction B by applying the Cope rearrangement mechanism to the identified reactant structure and considering the effect of heat."
    N_B = self.max_sc
    cot_agents_B = [LLMAgentBase(["thinking", "answer"], "CoT-SC Agent", model=self.node_model, temperature=0.5) for _ in range(N_B)]
    possible_answers_B = []
    thinkingmapping_B = {}
    answermapping_B = {}
    
    for i in range(N_B):
        thinking4, answer4 = await cot_agents_B[i]([taskInfo, thinking2, answer2], cot_sc_instruction_B, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_B[i].id}, evaluating products for reaction B, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_B.append(answer4.content)
        thinkingmapping_B[answer4.content] = thinking4
        answermapping_B[answer4.content] = answer4
    
    sub_tasks.append(f"Sub-task 4 output: possible answers for B: {possible_answers_B}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    debate_instruction = "Sub-task 5: Compare the evaluated products from subtasks 3 and 4 against the provided choices to determine which options match the expected products."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, possible_answers_A, possible_answers_B], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, possible_answers_A, possible_answers_B] + all_thinking[r-1] + all_answer[r-1]
                thinking5, answer5 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing products, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking[r].append(thinking5)
            all_answer[r].append(answer5)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_final, answer_final = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], "Sub-task 5: Make final decision on the products A and B.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final output, thinking: {thinking_final.content}; answer: {answer_final.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_final.content}; answer - {answer_final.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_final, answer_final, sub_tasks, agents)
    return final_answer