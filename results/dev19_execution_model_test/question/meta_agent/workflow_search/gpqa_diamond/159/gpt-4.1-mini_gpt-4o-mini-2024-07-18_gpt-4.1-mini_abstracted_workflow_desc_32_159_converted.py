async def forward_159(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage0 = "Sub-task 1: Extract and summarize the defining features of the aperture and diffraction setup, including the shape, apothem length, wavelength, incident light direction, and the limit as N approaches infinity. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_stage0 = []
    all_answer_stage0 = []
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instr_stage0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_stage0):
        thinking, answer = await agent([taskInfo], debate_instr_stage0, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, extracting and summarizing problem setup, thinking: {thinking.content}; answer: {answer.content}")
        all_thinking_stage0.append(thinking)
        all_answer_stage0.append(answer)
    final_decision_agent_stage0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_stage0([taskInfo] + all_thinking_stage0 + all_answer_stage0, "Sub-task 1: Extract and summarize problem setup. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage_0, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    debate_instr_stage1_1 = "Sub-task 1: Analyze the diffraction pattern characteristics for a regular polygonal aperture and understand the transition to a circular aperture as N approaches infinity, identifying the relevant diffraction minima conditions. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_stage1_1 = []
    all_answer_stage1_1 = []
    subtask_desc1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_stage1_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_stage1_1):
        thinking, answer = await agent([taskInfo, thinking0, answer0], debate_instr_stage1_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, stage_1.subtask_1, round 0, analyzing diffraction pattern, thinking: {thinking.content}; answer: {answer.content}")
        all_thinking_stage1_1.append(thinking)
        all_answer_stage1_1.append(answer)
    final_decision_agent_stage1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_stage1_1([taskInfo, thinking0, answer0] + all_thinking_stage1_1 + all_answer_stage1_1, "Sub-task 1: Analyze diffraction pattern and transition to circular aperture. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage_1.subtask_1, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_stage1_2 = "Sub-task 2: Compute the angular positions of the first two minima in the diffraction pattern for a circular aperture of radius a using the known Airy pattern formula and apply the small angle approximation."
    N_sc = self.max_sc
    cot_agents_stage1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage1_2 = []
    possible_thinkings_stage1_2 = []
    subtask_desc1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_stage1_2,
        "context": ["user query", thinking0, answer0, thinking1_1, answer1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_stage1_2[i]([taskInfo, thinking0, answer0, thinking1_1, answer1_1], cot_sc_instruction_stage1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1_2[i].id}, stage_1.subtask_2, considering angular minima, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_stage1_2.append(answer)
        possible_thinkings_stage1_2.append(thinking)
    final_decision_agent_stage1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_stage1_2([taskInfo, thinking0, answer0, thinking1_1, answer1_1] + possible_thinkings_stage1_2 + possible_answers_stage1_2, "Sub-task 2: Compute angular positions of first two minima and select most consistent answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage_1.subtask_2, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_stage2_1 = "Sub-task 1: Calculate the angular distance between the first two minima by subtracting their angular positions and express the result in terms of wavelength lambda and apothem length a."
    cot_agents_stage2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage2_1 = []
    possible_thinkings_stage2_1 = []
    subtask_desc2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_stage2_1,
        "context": ["user query", thinking1_2, answer1_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_stage2_1[i]([taskInfo, thinking1_2, answer1_2], cot_sc_instruction_stage2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2_1[i].id}, stage_2.subtask_1, calculating angular distance, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_stage2_1.append(answer)
        possible_thinkings_stage2_1.append(thinking)
    final_decision_agent_stage2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_stage2_1([taskInfo, thinking1_2, answer1_2] + possible_thinkings_stage2_1 + possible_answers_stage2_1, "Sub-task 1: Calculate angular distance between first two minima and select most consistent answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage_2.subtask_1, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_stage3_1 = "Sub-task 1: Compare the computed angular distance with the given answer choices and select the correct option that matches the theoretical result. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_stage3_1 = []
    all_answer_stage3_1 = []
    subtask_desc3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_stage3_1,
        "context": ["user query", thinking2_1, answer2_1],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_stage3_1):
        thinking, answer = await agent([taskInfo, thinking2_1, answer2_1], debate_instr_stage3_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, stage_3.subtask_1, round 0, comparing with choices, thinking: {thinking.content}; answer: {answer.content}")
        all_thinking_stage3_1.append(thinking)
        all_answer_stage3_1.append(answer)
    final_decision_agent_stage3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_stage3_1([taskInfo, thinking2_1, answer2_1] + all_thinking_stage3_1 + all_answer_stage3_1, "Sub-task 1: Compare computed angular distance with choices and select correct answer. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage_3.subtask_1, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Stage 3 Subtask 1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_1, answer3_1, sub_tasks, agents)
    return final_answer, logs
