async def forward_148(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze raw data (NMR and LC-MS) using Chain-of-Thought (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Analyze the 1H NMR spectrum data of the crude peptidic compound, "
        "focusing on the observation of two peaks corresponding to the same alpha-proton with similar chemical shifts and roughly equal integrals, "
        "and confirm that spin-spin coupling is ruled out as the cause of these duplicate peaks."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, NMR analysis, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze LC-MS data using Self-Consistency Chain-of-Thought (SC-CoT)
    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the LC-MS data of the crude compound at elevated temperature, "
        "noting the presence of two clearly defined peaks of equal intensities and confirming that both peaks have identical mass spectra consistent with the expected molecule."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, LC-MS analysis, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify and classify possible chemical or stereochemical explanations using Reflexion
    cot_reflect_instruction_3 = (
        "Sub-task 3: Identify and classify possible chemical or stereochemical explanations for the presence of two peaks in both NMR and LC-MS data "
        "that correspond to the same molecular mass and similar integrals, considering the options: mixture of enantiomers, mixture of diastereoisomers, "
        "double coupling during amide bond formation, or contamination with precursor."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                                  model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]

    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, classifying explanations, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], 
                                                "please review the classification of possible explanations and provide its limitations.", 
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining classification, thinking: {thinking3.content}; answer: {answer3.content}")

    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Evaluate likelihood of each explanation using Debate for subtasks 4 to 7
    debate_instruction = (
        "Evaluate the likelihood of each possible explanation for the observed NMR and LC-MS data: "
        "(1) mixture of enantiomers, (2) mixture of diastereoisomers, (3) double coupling during amide bond formation, "
        "and (4) contamination with precursor. Provide arguments for and against each option based on the data."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                  model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_debate = self.max_round

    all_thinking_debate = [[] for _ in range(N_max_debate)]
    all_answer_debate = [[] for _ in range(N_max_debate)]

    for r in range(N_max_debate):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3] + all_thinking_debate[r-1] + all_answer_debate[r-1]
                thinking_d, answer_d = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating explanations, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking_debate[r].append(thinking_d)
            all_answer_debate[r].append(answer_d)

    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                       model=self.node_model, temperature=0.0)
    thinking_final, answer_final = await final_decision_agent(
        [taskInfo] + all_thinking_debate[-1] + all_answer_debate[-1],
        "Sub-task 8: Integrate evaluations and make the final decision on the most likely explanation for the observed NMR and LC-MS data.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, making final decision, thinking: {thinking_final.content}; answer: {answer_final.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking_final.content}; answer - {answer_final.content}")
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_final, answer_final, sub_tasks, agents)
    return final_answer
