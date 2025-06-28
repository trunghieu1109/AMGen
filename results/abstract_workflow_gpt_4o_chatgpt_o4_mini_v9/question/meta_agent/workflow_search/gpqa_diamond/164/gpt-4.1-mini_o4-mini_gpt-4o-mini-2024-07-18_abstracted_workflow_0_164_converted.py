async def forward_164(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze and Classify Statements
    # Sub-task 1: Analyze the key chemical concepts involved
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given query context to identify the key chemical concepts involved: "
        "ethylene polymerization, homogeneous organometallic catalyst systems, polymer branching, and dual catalyst systems using only ethylene as monomer."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing key chemical concepts, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Classify the four statements by chemical and industrial relevance
    cot_sc_instruction_2 = (
        "Sub-task 2: Classify the four statements provided by the senior scientist according to their chemical and industrial relevance "
        "to the polymerization process described, focusing on catalyst types, activators, industrial implementation, and cost considerations."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, classifying statements, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Majority vote or most consistent answer
    answer2_final = Counter(possible_answers_2).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinkingmapping_2[answer2_final].content}; answer - {answer2_final}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Assess Each Statement
    # Sub-task 3: Assess dual catalyst system feasibility for regular branching
    cot_instruction_3 = (
        "Sub-task 3: Assess the role and feasibility of using a dual catalyst system to introduce regular branches in a polyethylene polymer backbone "
        "using only ethylene as the monomer, based on known polymer chemistry principles and catalyst behavior."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinkingmapping_2[answer2_final], answermapping_2[answer2_final]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, assessing dual catalyst system feasibility, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate claim about aluminum-based activators
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the specific claim that aluminum-based activators do not work for the essential additional reaction step in the dual catalyst system, "
        "considering known activator-catalyst interactions in ethylene polymerization."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinkingmapping_2[answer2_final], answermapping_2[answer2_final]], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating aluminum-based activators claim, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Assess validity of group VIa transition metal catalyst claim
    cot_instruction_5 = (
        "Sub-task 5: Assess the validity of the statement that group VIa transition metal catalysts can be used with specific activators "
        "to achieve the desired polymer branching in ethylene polymerization."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinkingmapping_2[answer2_final], answermapping_2[answer2_final]], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, assessing group VIa catalyst claim, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Evaluate industrial implementation claim
    cot_instruction_6 = (
        "Sub-task 6: Evaluate the statement regarding the industrial implementation of such combined catalyst systems in the US, "
        "verifying if such technology is currently in commercial use."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinkingmapping_2[answer2_final], answermapping_2[answer2_final]], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, evaluating industrial implementation claim, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Analyze noble metal catalyst cost claim
    cot_instruction_7 = (
        "Sub-task 7: Analyze the claim that certain noble metal catalysts can be used but are too expensive, focusing on the practicality and economic considerations "
        "of noble metal catalysts in ethylene polymerization for branching."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinkingmapping_2[answer2_final], answermapping_2[answer2_final]], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, analyzing noble metal catalyst cost claim, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Stage 2: Integrate Assessments
    # Sub-task 8: Integrate assessments to determine which statement is correct
    debate_instruction_8 = (
        "Sub-task 8: Integrate the assessments from subtasks 3 to 7 to determine which of the four statements is correct regarding the formation of a polymer with regular branches "
        "using only ethylene and a dual catalyst system."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]

    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                input_infos_8 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7]
            else:
                input_infos_8 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
            thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating assessments, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)

    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Make final decision on which statement is correct regarding polymer branching with dual catalyst system.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on correct statement, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
