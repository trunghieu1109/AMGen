async def forward_170(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = (
        "Sub-task 1: Extract and summarize the essential features of the six substances, their substituents, "
        "and the detailed reaction conditions relevant to electrophilic bromination and para-isomer formation. "
        "Explicitly clarify the meaning of 'weight fraction' (mass fraction vs. molar fraction) and specify the reaction environment, "
        "including catalyst type (e.g., FeBr3), solvent, and temperature, to ground subsequent analysis in realistic experimental settings. "
        "This avoids ambiguity and supports accurate interpretation of substituent effects."
    )
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, consider all possible cases of stage_1_subtask_1, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + possible_thinkings_1 + possible_answers_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct summary for the substances and reaction conditions.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_2 = (
        "Sub-task 2: Collect and justify the appropriate Hammett constants (σ_p) and other quantitative substituent parameters relevant to para substitution under Lewis acid catalysis conditions. "
        "Include a focused analysis of catalyst–substituent interactions, especially protonation or complexation effects on –COOH and –COOR groups, to correctly assess their relative deactivating strengths. "
        "This subtask addresses the previous error of misranking ester vs. acid by relying on inappropriate σ values and ignoring catalyst effects. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_max_2)]
    all_answer_2 = [[] for _ in range(N_max_2)]
    subtask_desc_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking_2[r].append(thinking2)
            all_answer_2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + all_thinking_2[-1] + all_answer_2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 1 of Stage 2: Analyze and classify the substituents on the benzene ring by separating their electronic effects (activating/deactivating) from steric effects influencing para-isomer formation. "
        "Use the quantitative data from stage_1.subtask_2 to support classification. Explicitly consider steric hindrance differences among substituents, including halogens and bulky groups, and how these affect regioselectivity and para-isomer yield. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking3_1, answer3_1 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking3_1, answer3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
            all_thinking_3_1[r].append(thinking3_1)
            all_answer_3_1[r].append(answer3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1(
        [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_3_1[-1] + all_answer_3_1[-1],
        "Sub-task 1 of Stage 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    debate_instruction_3_2 = (
        "Sub-task 2 of Stage 2: Predict the relative weight fractions of the para-isomer yields for each substituted benzene by integrating the electronic and steric classifications with the clarified reaction conditions and quantitative substituent parameters. "
        "Cross-validate predictions with known experimental trends or authoritative literature data where available. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3_1.content, answer3_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking3_2, answer3_2 = await agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos_3_2 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1] + all_thinking_3_2[r-1] + all_answer_3_2[r-1]
                thinking3_2, answer3_2 = await agent(input_infos_3_2, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
            all_thinking_3_2[r].append(thinking3_2)
            all_answer_3_2[r].append(answer3_2)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent_3_2(
        [taskInfo, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1] + all_thinking_3_2[-1] + all_answer_3_2[-1],
        "Sub-task 2 of Stage 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    debate_instruction_4 = (
        "Stage 3 Sub-task 1: Evaluate and prioritize the six substances based on the predicted para-isomer weight fractions to arrange them in order of increasing para-isomer yield. "
        "Critically assess the final ranking against all prior evidence, including catalyst effects, substituent parameters, steric considerations, and experimental data. "
        "Use a structured debate to resolve any conflicts or ambiguities and ensure the final order is robust and justified. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3_1.content, answer3_1.content, thinking3_2.content, answer3_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent(
                    [taskInfo, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1, thinking3_2, answer3_2],
                    debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1, thinking3_2, answer3_2] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4(
        [taskInfo, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1, thinking3_2, answer3_2] + all_thinking_4[-1] + all_answer_4[-1],
        "Stage 3 Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
