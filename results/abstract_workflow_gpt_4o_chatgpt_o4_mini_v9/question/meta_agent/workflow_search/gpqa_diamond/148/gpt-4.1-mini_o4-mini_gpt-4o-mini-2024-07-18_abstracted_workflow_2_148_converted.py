async def forward_148(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the 1H NMR spectrum data
    cot_instruction_1 = (
        "Sub-task 1: Analyze the 1H NMR spectrum data of the crude peptidic compound, "
        "focusing on the observation of two peaks corresponding to the same alpha-proton with similar chemical shifts and roughly equal integrals, "
        "and confirm that spin-spin coupling is ruled out as the cause of these duplicate peaks."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing 1H NMR spectrum, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze the LC-MS data of the crude compound at elevated temperature
    cot_instruction_2 = (
        "Sub-task 2: Analyze the LC-MS data of the crude compound at elevated temperature, "
        "noting the presence of two clearly defined peaks of equal intensities and confirming that both peaks have identical mass spectra consistent with the expected molecule."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing LC-MS data, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Integrate findings from 1H NMR and LC-MS analyses
    cot_reflect_instruction_3 = (
        "Sub-task 3: Integrate the findings from the 1H NMR and LC-MS analyses to characterize the nature of the two observed species, "
        "considering that both have the same molecular mass and similar NMR features but are distinct chromatographically."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, integrating NMR and LC-MS findings, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Critically evaluate the integration of NMR and LC-MS data and provide limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining integration, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Evaluate possible explanations

    # Sub-task 4: Evaluate possibility of enantiomers
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the possibility that the crude compound exists as a mixture of enantiomers, "
        "considering that enantiomers have identical physical properties in achiral environments and typically do not separate in standard LC-MS or show distinct NMR peaks without chiral agents."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating enantiomer possibility, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate possibility of diastereoisomers
    cot_instruction_5 = (
        "Sub-task 5: Evaluate the possibility that the crude compound exists as a mixture of diastereoisomers, "
        "which can have different physical properties, including distinct chromatographic behavior and NMR signals, consistent with the observed duplicate peaks and LC-MS separation."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, evaluating diastereoisomer possibility, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Assess likelihood of double coupling during amide-bond formation
    cot_instruction_6 = (
        "Sub-task 6: Assess the likelihood that double coupling occurred during the amide-bond forming reaction, "
        "which could cause unusual NMR patterns, and determine if this explains the observed duplicate alpha-proton peaks and LC-MS data."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, assessing double coupling, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Assess contamination with precursor compound
    cot_instruction_7 = (
        "Sub-task 7: Assess whether contamination with a precursor compound could explain the observed NMR and LC-MS data, "
        "considering the identical mass spectra and the chromatographic separation."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking3, answer3], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, assessing contamination, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Compare and prioritize explanations using Debate
    debate_instruction_8 = (
        "Sub-task 8: Compare and prioritize the evaluated explanations (enantiomers, diastereoisomers, double coupling, contamination) "
        "against the combined spectral and chromatographic data to identify the most likely explanation for the observations."
    )
    debate_roles = ["Pro-Enantiomer", "Pro-Diastereoisomer", "Pro-DoubleCoupling", "Pro-Contamination"]
    debate_agents_8 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in debate_roles
    ]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]

    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent(
                    [taskInfo, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7],
                    debate_instruction_8, r, is_sub_task=True
                )
            else:
                input_infos_8 = [taskInfo, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating explanations, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)

    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8(
        [taskInfo] + all_thinking8[-1] + all_answer8[-1],
        "Sub-task 8: Make final decision on the most likely explanation for the observations.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, deciding most likely explanation, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
