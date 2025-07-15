async def forward_182(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Analyze and classify the structural features of the starting compound 2-formyl-5-vinylcyclohex-3-enecarboxylic acid, explicitly identifying and counting all rings, pi bonds (double/triple bonds), and functional groups. Avoid omitting ring unsaturation in the IHD calculation by separately counting rings and pi bonds."
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, analyzing structure, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1 + possible_answers_1, "Sub-task 1: Synthesize and choose the most consistent structural analysis.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, calculate the initial index of hydrogen deficiency (IHD) of the starting compound using the standard formula (IHD = (2C + 2 - H)/2), explicitly including the contribution from rings and pi bonds. Ensure no unsaturation is overlooked."
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating initial IHD, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "Sub-task 2: Synthesize and choose the most consistent initial IHD calculation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3_1 = "Sub-task 1: Catalog and analyze the chemical transformations induced by red phosphorus and excess HI on each functional group present in the starting compound (formyl, vinyl, carboxylic acid, ring double bond). Explicitly distinguish which groups are reduced and which remain unaffected, based on mechanistic and literature evidence."
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_3_1 + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.",
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking3_1, answer3_1 = await agent([taskInfo, thinking1, answer1], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking1, answer1] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking3_1, answer3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing transformations, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
            all_thinking_3_1[r].append(thinking3_1)
            all_answer_3_1[r].append(answer3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking1, answer1] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 3.1: Synthesize and choose the most consistent chemical transformation analysis.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_sc_instruction_3_2 = "Sub-task 2: Predict the structure of the product after reaction with red phosphorus and excess HI by applying the transformations identified in Sub-task 3.1. Carefully consider the fate of the ring, ring double bond, vinyl group, and oxygen-containing groups, ensuring the ring structure and unsaturations are correctly represented."
    cot_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3_2 = []
    possible_thinkings_3_2 = []
    subtask_desc_3_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_3_2,
        "context": ["user query", thinking3_1, answer3_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3_2, answer3_2 = await cot_agents_3_2[i]([taskInfo, thinking3_1, answer3_1], cot_sc_instruction_3_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_2[i].id}, predicting product structure, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
        possible_answers_3_2.append(answer3_2)
        possible_thinkings_3_2.append(thinking3_2)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent_3_2([taskInfo, thinking3_1, answer3_1] + possible_thinkings_3_2 + possible_answers_3_2, "Sub-task 3.2: Synthesize and choose the most consistent product structure prediction.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    cot_sc_instruction_4 = "Sub-task 1: Calculate the index of hydrogen deficiency (IHD) of the predicted product structure, explicitly counting rings and pi bonds, and incorporating the changes from the reaction. Avoid assuming saturation of ring double bonds without evidence."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc_4 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking2, answer2, thinking3_2, answer3_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2, thinking3_2, answer3_2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, calculating product IHD, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking2, answer2, thinking3_2, answer3_2] + possible_thinkings_4 + possible_answers_4, "Sub-task 4: Synthesize and choose the most consistent product IHD calculation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = "Sub-task 1: Select the correct IHD value of the product from the given choices (0, 1, 3, 5) based on the computed IHD. Justify the selection by referencing the detailed structural analysis, reaction transformations, and IHD calculation. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4, answer4],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting final IHD, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking_5[r].append(thinking5)
            all_answer_5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
