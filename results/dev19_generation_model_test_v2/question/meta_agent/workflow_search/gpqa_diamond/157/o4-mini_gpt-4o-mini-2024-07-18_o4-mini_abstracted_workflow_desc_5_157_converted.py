async def forward_157(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: SC_CoT
    cot_sc_instruction = (
        "Sub-task 1: Extract and summarize the transcription factor’s activation pathway, the nature of mutations X (recessive LOF) and Y (heterozygous DN), and enumerate the four answer choices. "
        "Explicitly note past error: agents undervalued DN aggregation/sequestration and over-emphasized degradation."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_i, answer_i = await cot_agents[i](
            [taskInfo], cot_sc_instruction, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings.append(thinking_i)
        possible_answers.append(answer_i)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Given all the above thinking and answers, find the most consistent and correct summary of the problem and choices."
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1: Synthesize and choose the most consistent summary." + final_instr_1,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Debate - retrieve canonical DN mechanisms
    debate_instr = (
        "Given solutions to the problem from SC-CoT agent, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_instruction_2 = (
        "Sub-task 2: Retrieve and list canonical molecular mechanisms of dominant-negative mutations in dimerization domains, explicitly distinguishing between stable nonfunctional complex formation/aggregation/sequestration versus proteasomal degradation. "
        "Embed feedback reminder: avoid defaulting to degradation without evidence."
        + debate_instr
    )
    debate_agents_2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", "response of subtask_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent(
                    [taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True
                )
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(inputs, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2: Synthesize mechanisms." + final_instr_2,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Debate - map mutation Y
    debate_instruction_3 = (
        "Sub-task 3: Map mutation Y to the most plausible dominant-negative mechanism(s) based on its location in the dimerization domain and known domain behavior; generate a short list of candidate molecular phenotypes supported by evidence."
        + debate_instr
    )
    debate_agents_3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "response of subtask_2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent(
                    [taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True
                )
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(inputs, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3 = "Given all the above thinking and answers, reason over them carefully and provide a final list of candidate phenotypes."
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3: Synthesize candidate phenotypes." + final_instr_3,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Debate - select most likely phenotype
    debate_instruction_4 = (
        "Sub-task 4: Critically compare each enumerated answer choice against the candidate mechanisms, prioritizing aggregation/sequestration over degradation per canonical DN behavior, and select the single most likely molecular phenotype for mutation Y."
        + debate_instr
    )
    debate_agents_4 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "response of subtask_3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent(
                    [taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True
                )
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(inputs, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Given all the above thinking and answers, reason over them carefully and provide the final single phenotype answer."
    thinking4, answer4 = await final_decision_agent_4(
        [taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1],
        "Sub-task 4: Choose the final phenotype." + final_instr_4,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs