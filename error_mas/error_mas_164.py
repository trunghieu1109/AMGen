async def forward_164(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Analyze and classify the key elements from the query: catalyst types (group VIa transition metals, noble metals), activators (aluminum-based and others), industrial implementation status, and the mechanism of branching using only ethylene."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing key elements, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers.append(answer1)
        possible_thinkings.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1: Synthesize and choose the most consistent classification of key elements from the query.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr_template = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_roles = self.debate_role
    N_max = self.max_round

    def run_debate_subtask(subtask_id, instruction, context_inputs):
        async def inner():
            debate_instruction = f"{instruction} " + debate_instr_template
            debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
            all_thinking = [[] for _ in range(N_max)]
            all_answer = [[] for _ in range(N_max)]
            subtask_desc = {
                "subtask_id": subtask_id,
                "instruction": debate_instruction,
                "context": context_inputs,
                "agent_collaboration": "Debate"
            }
            for r in range(N_max):
                for i, agent in enumerate(debate_agents):
                    if r == 0:
                        thinking, answer = await agent(context_inputs, debate_instruction, r, is_sub_task=True)
                    else:
                        input_infos = context_inputs + all_thinking[r-1] + all_answer[r-1]
                        thinking, answer = await agent(input_infos, debate_instruction, r, is_sub_task=True)
                    agents.append(f"Debate agent {agent.id}, round {r}, subtask {subtask_id}, thinking: {thinking.content}; answer: {answer.content}")
                    all_thinking[r].append(thinking)
                    all_answer[r].append(answer)
            final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
            thinking_final, answer_final = await final_decision_agent(context_inputs + all_thinking[-1] + all_answer[-1], f"{subtask_id}: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
            agents.append(f"Final Decision agent, subtask {subtask_id}, thinking: {thinking_final.content}; answer: {answer_final.content}")
            sub_tasks.append(f"{subtask_id} output: thinking - {thinking_final.content}; answer - {answer_final.content}")
            subtask_desc['response'] = {
                "thinking": thinking_final,
                "answer": answer_final
            }
            logs.append(subtask_desc)
            print(f"Step {len(sub_tasks)+1}: ", sub_tasks[-1])
            return thinking_final, answer_final
        return inner

    thinking2, answer2 = await run_debate_subtask(
        "subtask_2",
        "Sub-task 2: Evaluate the feasibility and industrial implementation of dual catalyst systems for ethylene polymerization producing branched polymers using only ethylene, focusing on the validity of the statement about industrial scale use in the US.",
        [taskInfo, thinking1, answer1]
    )()

    thinking3, answer3 = await run_debate_subtask(
        "subtask_3",
        "Sub-task 3: Assess the role and effectiveness of aluminum-based activators in the essential additional reaction step for branching, determining if they indeed do not work as claimed.",
        [taskInfo, thinking1, answer1]
    )()

    thinking4, answer4 = await run_debate_subtask(
        "subtask_4",
        "Sub-task 4: Analyze the use of group VIa transition metal catalysts with specific activators for introducing regular branches in polyethylene from ethylene alone, verifying the correctness of this statement.",
        [taskInfo, thinking1, answer1]
    )()

    thinking5, answer5 = await run_debate_subtask(
        "subtask_5",
        "Sub-task 5: Evaluate the statement regarding the use and cost implications of certain noble metal catalysts for the branching reaction step, considering industrial practicality and expense.",
        [taskInfo, thinking1, answer1]
    )()

    final_answer = await self.make_final_answer(
        [thinking2, thinking3, thinking4, thinking5],
        [answer2, answer3, answer4, answer5],
        sub_tasks,
        agents
    )
    return final_answer, logs
