async def forward_155(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage1_1 = (
        "Sub-task 1: Perform a detailed stereochemical analysis of the epoxidation reaction on (E)- and (Z)-oct-4-ene with mCPBA, "
        "explicitly checking for molecular symmetry and determining whether the products are meso compounds or single enantiomers rather than racemic mixtures. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage1_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_stage1_1 = self.max_round

    all_thinking_stage1_1 = [[] for _ in range(N_max_stage1_1)]
    all_answer_stage1_1 = [[] for _ in range(N_max_stage1_1)]

    subtask_desc1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_stage1_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_stage1_1):
        for i, agent in enumerate(debate_agents_stage1_1):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr_stage1_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking_stage1_1[r-1] + all_answer_stage1_1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage1_1[r].append(thinking)
            all_answer_stage1_1[r].append(answer)

    final_decision_agent_stage1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_stage1_1(
        [taskInfo] + all_thinking_stage1_1[-1] + all_answer_stage1_1[-1],
        "Sub-task 1: Perform a detailed stereochemical analysis of the epoxidation reaction on (E)- and (Z)-oct-4-ene with mCPBA, explicitly checking for molecular symmetry and determining whether the products are meso compounds or single enantiomers rather than racemic mixtures. Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, stage_1.subtask_1, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc1_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_stage1_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, enumerate and characterize the stereoisomeric epoxide products formed from each alkene isomer after epoxidation and aqueous acid treatment, "
        "including counting diastereomers and enantiomers correctly, avoiding overcounting."
    )
    N_sc = self.max_sc
    cot_agents_stage1_2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc)
    ]

    possible_answers_stage1_2 = []
    possible_thinkings_stage1_2 = []

    subtask_desc1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_stage1_2,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(N_sc):
        thinking, answer = await cot_agents_stage1_2[i](
            [taskInfo, thinking1_1, answer1_1], cot_sc_instruction_stage1_2, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents_stage1_2[i].id}, enumerating stereoisomers, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_stage1_2.append(answer)
        possible_thinkings_stage1_2.append(thinking)

    final_decision_agent_stage1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_stage1_2(
        [taskInfo, thinking1_1, answer1_1] + possible_thinkings_stage1_2 + possible_answers_stage1_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct enumeration of stereoisomeric epoxide products formed from each alkene isomer after epoxidation and aqueous acid treatment.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, stage_1.subtask_2, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc1_2)
    print("Step 2: ", sub_tasks[-1])

    reflect_instruction_stage2_1 = (
        "Sub-task 1: Combine the stereoisomeric product sets from both (E)- and (Z)-oct-4-ene reactions to enumerate the total distinct stereoisomers present in the combined mixture. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_stage2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round

    cot_inputs_stage2_1 = [taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2]

    subtask_desc2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": reflect_instruction_stage2_1,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "agent_collaboration": "Reflexion"
    }

    thinking2_1, answer2_1 = await cot_agent_stage2_1(cot_inputs_stage2_1, reflect_instruction_stage2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2_1.id}, combining stereoisomers, thinking: {thinking2_1.content}; answer: {answer2_1.content}")

    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_stage2_1(
            [taskInfo, thinking2_1, answer2_1],
            "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
            i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent_stage2_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_stage2_1.extend([thinking2_1, answer2_1, feedback])
        thinking2_1, answer2_1 = await cot_agent_stage2_1(cot_inputs_stage2_1, reflect_instruction_stage2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2_1.id}, refining combination, thinking: {thinking2_1.content}; answer: {answer2_1.content}")

    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_stage3_1 = (
        "Sub-task 1: Predict the chromatographic behavior of the combined product mixture on a standard (achiral) reverse-phase HPLC column, "
        "focusing on the number of peaks expected based on the separation of diastereomers and the co-elution of enantiomers. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage3_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_stage3_1 = self.max_round

    all_thinking_stage3_1 = [[] for _ in range(N_max_stage3_1)]
    all_answer_stage3_1 = [[] for _ in range(N_max_stage3_1)]

    subtask_desc3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_stage3_1,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_stage3_1):
        for i, agent in enumerate(debate_agents_stage3_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2_1, answer2_1], debate_instr_stage3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2_1, answer2_1] + all_thinking_stage3_1[r-1] + all_answer_stage3_1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage3_1[r].append(thinking)
            all_answer_stage3_1[r].append(answer)

    final_decision_agent_stage3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_stage3_1(
        [taskInfo, thinking2_1, answer2_1] + all_thinking_stage3_1[-1] + all_answer_stage3_1[-1],
        "Sub-task 1: Predict chromatographic behavior on achiral HPLC. Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, stage_3.subtask_1, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_stage3_2 = (
        "Sub-task 2: Predict the chromatographic behavior of the combined product mixture on a chiral HPLC column, "
        "focusing on the number of peaks expected based on the separation of both diastereomers and enantiomers. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage3_2 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_stage3_2 = self.max_round

    all_thinking_stage3_2 = [[] for _ in range(N_max_stage3_2)]
    all_answer_stage3_2 = [[] for _ in range(N_max_stage3_2)]

    subtask_desc3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instr_stage3_2,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_stage3_2):
        for i, agent in enumerate(debate_agents_stage3_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2_1, answer2_1], debate_instr_stage3_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2_1, answer2_1] + all_thinking_stage3_2[r-1] + all_answer_stage3_2[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage3_2[r].append(thinking)
            all_answer_stage3_2[r].append(answer)

    final_decision_agent_stage3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent_stage3_2(
        [taskInfo, thinking2_1, answer2_1] + all_thinking_stage3_2[-1] + all_answer_stage3_2[-1],
        "Sub-task 2: Predict chromatographic behavior on chiral HPLC. Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, stage_3.subtask_2, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc3_2)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(
        [thinking3_1, thinking3_2], [answer3_1, answer3_2], sub_tasks, agents
    )
    return final_answer, logs
