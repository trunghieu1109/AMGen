async def forward_78(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze molecular formula of Compound X (C11H12O)
    cot_instruction_1 = (
        "Sub-task 1: Analyze the molecular formula C11H12O of Compound X to determine possible structural features such as degree of unsaturation, "
        "presence of functional groups, and overall molecular framework relevant to the reaction conditions and NMR data."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing molecular formula, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze reaction conditions
    cot_instruction_2 = (
        "Sub-task 2: Based on the molecular formula analysis from Sub-task 1, analyze the reaction conditions: reaction of Compound X with 1,4-diazabicyclo[2.2.2]octane in a nonpolar solvent at 403 K for 1 day, "
        "to infer possible reaction types or transformations that Compound X might undergo under these conditions."
    )
    N_sc = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing reaction conditions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer (majority vote)
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_content = thinkingmapping_2[answer2_content].content
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_content}; answer - {answer2_content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze 1H NMR spectral data
    cot_instruction_3 = (
        "Sub-task 3: Analyze the provided 1H NMR spectral data (chemical shifts, multiplicities, integration) of the product to identify key proton environments and their possible structural implications, "
        "using insights from Sub-tasks 1 and 2."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answermapping_2[answer2_content]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing 1H NMR data, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze 13C NMR spectral data
    cot_instruction_4 = (
        "Sub-task 4: Analyze the provided 13C NMR spectral data (chemical shifts and number of carbons) of the product to identify key carbon environments and their possible structural implications, "
        "using insights from Sub-tasks 1 and 2."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking2, answermapping_2[answer2_content]], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing 13C NMR data, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Correlate and Select

    # Sub-task 5: Correlate 1H and 13C NMR data with molecular formula and reaction conditions
    debate_instruction_5 = (
        "Sub-task 5: Correlate the 1H and 13C NMR data with the molecular formula and reaction conditions to propose possible structural motifs or substructures present in the product. "
        "Debate among agents to evaluate different structural interpretations based on the spectral data and reaction context."
    )
    debate_roles = ["Spectral Analyst", "Synthetic Chemist"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round if hasattr(self, 'max_round') else 3
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent(
                    [taskInfo, thinking3, answer3, thinking4, answer4, thinking1, answer1, thinking2, answermapping_2[answer2_content]],
                    debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking1, answer1, thinking2, answermapping_2[answer2_content]] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, correlating NMR data and molecular info, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Make final decision on possible structural motifs based on debate.",
        is_sub_task=True)
    agents.append(f"Final Decision agent, deciding structural motifs, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare proposed motifs with candidate compounds
    cot_instruction_6 = (
        "Sub-task 6: Compare the proposed structural motifs and NMR features with the four given candidate compounds: 2-styrylepoxide, 2-(4-methylstyryl)oxirane, "
        "2-(1-phenylprop-1-en-2-yl)oxirane, and 2-methyl-3-styryloxirane, to evaluate which candidate best fits the spectral data and reaction context."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing candidates, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Select most likely identity of Compound X
    cot_reflect_instruction_7 = (
        "Sub-task 7: Based on comprehensive analysis of molecular formula, reaction conditions, NMR data, and candidate comparison, select the most likely identity of Compound X. "
        "Provide a clear rationale for the selection."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round if hasattr(self, 'max_round') else 3
    cot_inputs_7 = [taskInfo, thinking1, answer1, thinking2, answermapping_2[answer2_content], thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, selecting identity, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7],
                                                "Review the selection of Compound X identity and provide limitations or confirm correctness.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining selection, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
