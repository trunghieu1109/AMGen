async def forward_164(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = "Sub-task 1: Extract and classify the key elements from the query, including catalyst types, activators, polymer structure goals, and industrial context."
    N_sc_0 = self.max_sc
    cot_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0):
        thinking0, answer0 = await cot_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, extracting key elements, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, "Sub-task 1: Synthesize and choose the most consistent extraction and classification of key elements.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_template = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."

    debate_subtasks = [
        {
            "id": "stage_1.subtask_1",
            "objective": "Evaluate the industrial feasibility and implementation status of combined dual catalyst systems for ethylene polymerization with branching.",
            "statement": "Such combined systems are already implemented on an industrial scale in the US.",
            "context_key": "industrial feasibility"
        },
        {
            "id": "stage_1.subtask_2",
            "objective": "Assess the chemical compatibility and effectiveness of aluminum-based activators in the additional reaction step for branching.",
            "statement": "Aluminum-based activators do not work for the essential additional reaction step.",
            "context_key": "aluminum activators"
        },
        {
            "id": "stage_1.subtask_3",
            "objective": "Analyze the suitability of group VIa transition metal catalysts with specific activators for introducing regular branches in polyethylene using only ethylene monomer.",
            "statement": "One can use a catalyst of a group VIa transition metal in combination with specific activators.",
            "context_key": "group VIa catalysts"
        },
        {
            "id": "stage_1.subtask_4",
            "objective": "Evaluate the practicality and cost implications of using certain noble metal catalysts for the branching reaction step in ethylene polymerization.",
            "statement": "Certain noble metal catalysts can be used but are too expensive.",
            "context_key": "noble metal catalysts"
        }
    ]

    debate_role = self.debate_role
    max_round = self.max_round

    stage1_results = {}

    for subtask in debate_subtasks:
        debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_role]
        all_thinking = [[] for _ in range(max_round)]
        all_answer = [[] for _ in range(max_round)]
        subtask_desc = {
            "subtask_id": subtask["id"],
            "instruction": f"Sub-task: {subtask['objective']} Statement: '{subtask['statement']}'. " + debate_instruction_template,
            "context": ["user query", thinking0, answer0],
            "agent_collaboration": "Debate"
        }
        for r in range(max_round):
            for i, agent in enumerate(debate_agents):
                if r == 0:
                    thinking_d, answer_d = await agent([taskInfo, thinking0, answer0], subtask_desc["instruction"], r, is_sub_task=True)
                else:
                    input_infos = [taskInfo, thinking0, answer0] + all_thinking[r-1] + all_answer[r-1]
                    thinking_d, answer_d = await agent(input_infos, subtask_desc["instruction"], r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, subtask {subtask['id']}, thinking: {thinking_d.content}; answer: {answer_d.content}")
                all_thinking[r].append(thinking_d)
                all_answer[r].append(answer_d)
        final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking_final, answer_final = await final_decision_agent([taskInfo, thinking0, answer0] + all_thinking[-1] + all_answer[-1], f"Sub-task: {subtask['objective']} Statement: '{subtask['statement']}'. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
        agents.append(f"Final Decision agent for {subtask['id']}, thinking: {thinking_final.content}; answer: {answer_final.content}")
        sub_tasks.append(f"{subtask['id']} output: thinking - {thinking_final.content}; answer - {answer_final.content}")
        subtask_desc['response'] = {"thinking": thinking_final, "answer": answer_final}
        logs.append(subtask_desc)
        print(f"Step {len(sub_tasks)+1}: ", sub_tasks[-1])
        stage1_results[subtask['id']] = (thinking_final, answer_final)

    cot_sc_instruction_2 = "Sub-task: Integrate the evaluations from Stage 1 to determine which of the four statements is correct regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system."
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking0, answer0] + [v[0] for v in stage1_results.values()] + [v[1] for v in stage1_results.values()],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking0, answer0] + [v[0] for v in stage1_results.values()] + [v[1] for v in stage1_results.values()], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, integrating evaluations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking0, answer0] + possible_thinkings_2 + possible_answers_2, "Sub-task: Synthesize and choose the most consistent and correct solution for the final answer.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step final: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer, logs
