async def forward_54(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 1: Analyze and classify spectral features of the unknown compound
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given 1H NMR data (chemical shifts, multiplicities, coupling constants, and integration) "
        "to identify and classify the defining spectral features of the unknown compound, including the number and types of protons and their coupling patterns. "
        "Data: 7.0 ppm (1H, d, J=16.0 Hz), 5.5 ppm (1H, dq), 2.1 ppm (3H, s), 1.6 ppm (3H, d)."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing unknown compound NMR data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Stage 1: Review expected NMR features of candidate compounds with Self-Consistency CoT
    cot_sc_instruction_2 = (
        "Sub-task 2: Review the structural characteristics and typical 1H NMR spectral features of the four candidate compounds: "
        "Trans-propenyl acetate, Cis-propenyl acetate, Trans-butenyl acetate, and Cis-butenyl acetate. Focus on expected chemical shifts, coupling constants, and multiplicities for each isomer. "
        "Use self-consistency to consider multiple plausible spectral patterns for each isomer."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, reviewing candidate NMR features, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by frequency
    answer2_final = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 2: Compare unknown NMR features with candidate features and evaluate coupling constants
    cot_reflect_instruction_3 = (
        "Sub-task 3: Compare the analyzed NMR features of the unknown compound from Sub-task 1 with the expected NMR features of each candidate compound from Sub-task 2, "
        "identifying consistencies and discrepancies to narrow down possible matches."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2_final, answermapping_2[answer2_final]], cot_reflect_instruction_3, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, comparing unknown and candidate NMR features, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Sub-task 4: Evaluate coupling constants and multiplicities to distinguish cis/trans and propenyl/butenyl
    cot_reflect_instruction_4 = (
        "Sub-task 4: Evaluate the coupling constants (especially the J = 16.0 Hz doublet) and multiplicities in the unknown compound's NMR data to distinguish between cis and trans isomers "
        "and between propenyl and butenyl acetate structures, using known NMR coupling patterns for alkenes and esters."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking2_final, answermapping_2[answer2_final], thinking3, answer3], cot_reflect_instruction_4, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, evaluating coupling constants and multiplicities, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    # Sub-task 5: Integrate all spectral evidence and select the most likely identity using Debate
    debate_instruction_5 = (
        "Sub-task 5: Integrate all spectral evidence and reasoning from previous subtasks to select the most likely identity of the unknown compound among the four choices: "
        "Trans-propenyl acetate, Cis-propenyl acetate, Cis-butenyl acetate, or Trans-butenyl acetate."
    )
    debate_roles = ["Pro-trans isomer advocate", "Pro-cis isomer advocate"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent(
                    [taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating identity, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the unknown compound identity.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final identity, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
