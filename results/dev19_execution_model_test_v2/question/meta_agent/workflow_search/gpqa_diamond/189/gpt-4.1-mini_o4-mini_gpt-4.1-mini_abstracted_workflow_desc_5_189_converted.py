async def forward_189(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage1 = "Sub-task 1: Extract and summarize the defining chemical features of the nucleophiles and reaction context from the query, including their structures, charges, and the aqueous environment. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_stage1 = []
    all_answer_stage1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_stage1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_stage1):
        thinking1, answer1 = await agent([taskInfo], debate_instr_stage1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, extracting chemical features, thinking: {thinking1.content}; answer: {answer1.content}")
        all_thinking_stage1.append(thinking1)
        all_answer_stage1.append(answer1)
    final_decision_agent_stage1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_stage1([taskInfo] + all_thinking_stage1 + all_answer_stage1, "Sub-task 1: Synthesize and finalize the chemical features extraction." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_stage2 = "Sub-task 2: Based on the output from Sub-task 1, analyze and classify the nucleophiles based on their intrinsic nucleophilicity factors such as charge, electronegativity, resonance stabilization, and solvation effects in aqueous solution."
    N = self.max_sc
    cot_agents_stage2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_stage2 = []
    possible_thinkings_stage2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_stage2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_stage2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_stage2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2[i].id}, analyzing nucleophilicity, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_stage2.append(answer2)
        possible_thinkings_stage2.append(thinking2)
    final_decision_agent_stage2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_stage2([taskInfo, thinking1, answer1] + possible_thinkings_stage2 + possible_answers_stage2, "Sub-task 2: Synthesize and choose the most consistent nucleophile classification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_stage3 = "Sub-task 3: Transform the classified nucleophiles into possible reactivity orders by applying chemical principles of nucleophilicity in water, considering solvation and resonance effects. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_stage3 = []
    all_answer_stage3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_stage3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_stage3):
        thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_stage3, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, transforming classification to reactivity order, thinking: {thinking3.content}; answer: {answer3.content}")
        all_thinking_stage3.append(thinking3)
        all_answer_stage3.append(answer3)
    final_decision_agent_stage3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_stage3([taskInfo, thinking2, answer2] + all_thinking_stage3 + all_answer_stage3, "Sub-task 3: Synthesize and finalize nucleophile reactivity orders.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_stage4 = "Sub-task 4: Evaluate and prioritize the generated nucleophile reactivity orders against the provided multiple-choice options to select the most chemically accurate ranking. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_stage4 = []
    all_answer_stage4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr_stage4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_stage4):
        thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instr_stage4, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, evaluating and prioritizing reactivity orders, thinking: {thinking4.content}; answer: {answer4.content}")
        all_thinking_stage4.append(thinking4)
        all_answer_stage4.append(answer4)
    final_decision_agent_stage4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_stage4([taskInfo, thinking3, answer3] + all_thinking_stage4 + all_answer_stage4, "Sub-task 4: Select the most chemically accurate nucleophile reactivity ranking from the given choices.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
