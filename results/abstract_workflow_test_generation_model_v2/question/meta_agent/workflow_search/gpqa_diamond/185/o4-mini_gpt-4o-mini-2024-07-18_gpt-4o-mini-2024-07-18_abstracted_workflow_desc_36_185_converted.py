async def forward_185(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Generate a detailed, atom-indexed SMILES or adjacency list representation of (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene, including explicit numbering of all atoms and marking stereochemical configurations."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc = {"subtask_id":"subtask_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, generating structural representation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Perform at least three independent [3,3]-sigmatropic Cope rearrangement mappings on the indexed structure from Sub-task 1, providing for each mapping: identification of migrating bond and atom fragments, a bullet-point breakdown of bond-breaking and bond-forming events with electron flow arrows, and notation of any predicted stereochemical changes."
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking","answer"], "SC CoT Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await sc_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"SC-CoT agent {sc_agents[i].id}, mapping rearrangement {i+1}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinking_map[answer2.content] = thinking2
        answer_map[answer2.content] = answer2
    chosen = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_map[chosen]
    answer2 = answer_map[chosen]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc)
    print("Step 2: ", sub_tasks[-1])
    cot_reflect_instruction = "Sub-task 3: Enumerate relevant transition-state conformers for the Cope rearrangement, assess their relative feasibility, and determine the stereochemical outcome at original and newly formed stereocenters, confirming configuration retention or inversion."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    subtask_desc = {"subtask_id":"subtask_3","instruction":cot_reflect_instruction,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"Reflexion"}
    thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2], cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, initial conformer enumeration and stereochemistry, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "Please review the conformer feasibility and stereochemical assignments and provide feedback.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2, thinking3, answer3, feedback], cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refined conformer analysis and stereochemistry, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction = "Sub-task 4: Compare the fully specified predicted product's connectivity and stereochemistry from Sub-task 3 with the provided answer choices and select the matching letter (A, B, C, or D) with a brief justification."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc = {"subtask_id":"subtask_4","instruction":cot_instruction,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"CoT"}
    thinking4, answer4 = await cot_agent([taskInfo, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, comparing to choices, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs