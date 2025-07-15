async def forward_177(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage0 = "Sub-task 1: Extract and summarize the defining features of the problem, including the given Lagrangian, definitions of fields and operators, and known standard mass dimensions of fields in 4D QFT. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
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
        thinking0, answer0 = await agent([taskInfo], debate_instr_stage0, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, extracting and summarizing problem features, thinking: {thinking0.content}; answer: {answer0.content}")
        all_thinking_stage0.append(thinking0)
        all_answer_stage0.append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_stage0 + all_answer_stage0, "Sub-task 1: Extract and summarize problem features. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 0, calculating final summary, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Stage 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_stage1 = "Sub-task 1: Analyze the mass dimensions of each component in the interaction term (psi-bar sigma_mu_nu psi F^{mu nu}) and determine the mass dimension of the coupling constant kappa by enforcing the total Lagrangian dimension to be 4."
    N = self.max_sc
    cot_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_stage1 = []
    possible_thinkings_stage1 = []
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_stage1,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents_stage1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_stage1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1[i].id}, analyzing mass dimensions, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_stage1.append(answer1)
        possible_thinkings_stage1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings_stage1 + possible_answers_stage1, "Sub-task 2: Synthesize and choose the most consistent and correct solution for the mass dimension of kappa.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 1, calculating mass dimension of kappa, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Stage 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr_stage2 = "Sub-task 1: Evaluate the renormalizability of the theory based on the mass dimension of kappa and select the correct answer choice from the given options. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_stage2 = []
    all_answer_stage2 = []
    subtask_desc2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_stage2,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_stage2):
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instr_stage2, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, evaluating renormalizability, thinking: {thinking2.content}; answer: {answer2.content}")
        all_thinking_stage2.append(thinking2)
        all_answer_stage2.append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking_stage2 + all_answer_stage2, "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent stage 2, calculating final answer, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Stage 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer, logs
